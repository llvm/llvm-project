====================
``<rcu>`` Design
====================

.. contents::
   :local:
   :depth: 2


Introduction
============

This is the C++ paper `Read-Copy Update (RCU) <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2545r4.pdf>`__.
There are three designs discussed in the paper `Supplementary Material for User-Level Implementations of
Read-Copy Update <http://www.rdrop.com/users/paulmck/RCU/urcu-supp-accepted.2011.08.30a.pdf>`__

- Quiescent-State-Based Reclamation RCU
- General-Purpose RCU
- Low-Overhead RCU Via Signal Handling

libc++ adopted a variation that closely resembles the "General-Purpose RCU" design in the above paper.


Background Information
======================

`rcu` is an alternative to read-write locks and it is suitable for read-mostly data structures.
Here is an example usage ::

  struct Data /* members */ ;
  std::atomic<Data*> data_;

  // ==============================================
  template <typename Func>
  Result reader_op(Func fn) {
    std::scoped_lock l(std::rcu_default_domain());
    Data* p = data_;
    // fn should not block too long or call
    // rcu_synchronize(), rcu_barrier(), or
    // rcu_retire(), directly or indirectly
    return fn(p);
 }

  // ==============================================
  // May be called concurrently with reader_op
  void update(Data* newdata) {
    Data* olddata = data_.exchange(newdata);
    std::rcu_retire(olddata); // reclaim *olddata when safe
  }

There are several key details that an implementation of `rcu` must address:

- On the reader side, `rcu_domain::lock` and `rcu_domain::unlock` must not block the thread while the writer thread
  is performing updates.

- On the writer side, `retire` -ing an object in principle should not block at all. However, it is conforming to
  evaluate the `deleter` s inside `run_retire` .

- On the writer side, `rcu_synchronize` should block until at least all the existing readers exit their critical sections
  via `rcu_domain::unlock` . Note that this is a key difference between `rcu` and read-write locks:
  In case a late reader enters the critical section after the writer thread has called `rcu_synchronize` and waiting, the
  writer thread does not need to wait for the late reader to exit the critical section.

- On the writer side, `run_barrier` should block until all the `retired` objects that happen before the `run_barrier` call are reclaimed.

- The threads that are using `rcu` must be known by the `rcu` implementation states.


Adopted Design
==============

The core idea of `rcu` can be described in this simple image from lwn.net

.. image:: https://static.lwn.net/images/ns/kernel/rcu/GracePeriodGood.png

- Each row is a thread. The last row is the writer thread and the rows above are the reader threads.
- Each "Reader" block represents a critical section, which starts with `rcu_domain::lock` and ends with `rcu_domain::unlock` .
- When `run_retire` is called from the writer thread, it starts the "Removal" block.
- When `run_synchronize` is called from the writer thread, it starts the "Grace Period" block. We need to wait until all the 
  "Reader" blocks that started before the "Grace Period" started, to exit via `rcu_domain::unlock`, then we can end the "Grace Period".
  Note that the "Grace Period" ends after the 4th row's "Reader" block ends. Also note that the "Grace Period" does not need to wait
  for the late "Reader" blocks.
- After the "Grace Period" ends, we can start the "Reclaim" block to reclaim the retired objects.

Libc++ adopted a design that closely resembles the "General-Purpose RCU" design in the above paper.

Some key details of this design are:

- There is a global state which has two phases and it flips between the two phases.
- Each thread stores its state: whether there is a reader in the critical section, and which phase it was when it entered the critical section.
- When `run_synchronize` is called, it

  - flips the global state to the next phase
  - Going through a grace period: waits until all the threads that are in the critical section with the previous phase to exit the critical section.
  - flips the global state back to the original phase
  - Going through another grace period: waits until all the threads that are in the critical section ith the next phase to exit the critical section.

When `run_synchronize` returns, we can be sure that all the readers that were in the critical section before `run_synchronize` are now out of the critical section.
The paper explains why we need to wait two phases instead of just one phase in detail. The key point is that, if we only wait for the readers in the previous phase to exit,
there might be a late reader that enters the critical section after we flip the global state and before we wait for the previous phase's readers to exit.

Implementation Terminology
---------------------------

- reader: A thread that calls `rcu_domain::lock` and `rcu_domain::unlock` to enter and exit the critical section.

- quiescent state: When a thread is *not* in an RCU read-side critical section (between `lock` and `unlock`),
  it is in a quiescent state.

- grace period: Any time period during which every thread has been in at least one quiescent state is a grace period.
  The grace period cannot end until all pre-existing readers have exited their critical sections. (see the image above)

Implementation Details
----------------------

`thread_local_container`
~~~~~~~~~~~~~~~~~~~~~~~~

A helper class that manages thread local objects. It provides APIs to get the `thread_local` object for the current thread, and to iterate through all the `thread_local` objects from all threads.
It only contains `static` member variables and functions.

It uses `mutex` to protect the list of all object pointers. Since the list is only modified when a thread first calls `get_current_thread_instance` to register itself, and when a thread exits, 
the contention on the `mutex` should be low.

TODO: We need to replace `mutex` as all non allocating APIs in `rcu` are designed to be `noexcept` and `mutex` can throw.

`reader_states`
~~~~~~~~~~~~~~~

It defines the state of each reader thread. The state is essentially a pair of `(phase, lock_nested_level)` merged into a single `uint16_t` . 

- `phase` is a single bit as the global state only has two phases. This `phase` is the phase of the global state when the current thread enters the critical section.
- `lock_nested_level` is the nested level of the `rcu_domain::lock` . As you can call `rcu_domain::lock` multiple times in the same thread, we need to keep track of the nested level.
  When `lock_nested_level` is `0`, it means the current thread is in a quiescent state (not in a critical section).

The `reader_states` class is also a `static` helper class which only provides static definitions and functions, instead of the wrapper of the `uint16_t` state. This is because we need
the `atomic` integer API, e.g. `fetch_or`. An `atomic` of a wrapper struct won't have these APIs.

`rcu_singly_list_view`
~~~~~~~~~~~~~~~~~~~~~~

A helper class that provides a view of the intrusive singly linked list of `__rcu_node` s. It also stores the back pointer of the list to make the push back operation more efficient.
It provides APIs to push a node to the back of the list, splice from another list and to iterate through the list.

For the derived class of `rcu_obj_base`, the `retire` is `noexcept`, which means that no memory allocation can happen when pushing it to the retired callback queue.

`rcu_domain_impl`
~~~~~~~~~~~~~~~~~

This is the main class that implements the `rcu` logic. It contains

- `std::atomic<reader_states::state_type> global_reader_phase_` : the global state that flips between two phases. The readers will record the phase when they enter the critical section, and the writer will flip the global state when it calls `run_synchronize` to start a new grace period.

- `std::mutex grace_period_mutex_` : If we have multiple writer threads calling `run_synchronize` concurrenly, we need to make sure only one of them is performing the phase flipping and deleter queue draining.
  TODO: `mutex` can throw, we need to consider how to replace it.

- `std::atomic<bool> grace_period_waiting_flag_` : This flag is used to sleep/wake up the writer thread that is waiting for the grace period to end.

- `std::mutex retire_queue_mutex_` and  `rcu_singly_list_view __retired_callback_queue_` : This queue stores all the retired callbacks that are waiting for the grace period to end.
  TODO: `mutex` can throw, we need to consider how to replace it.

- `rcu_singly_list_view callbacks_phase_1_` and `rcu_singly_list_view callbacks_phase_2_` : These two queues are used to let the queued retired callbacks to go through two grace periods before invocation. No additional synchronization is needed for these two queues as they are only processed when the writer thread is holding the `grace_period_mutex_` .

The domain has few operations:

`lock`
^^^^^^

- If current was in quiescent state, we need to record the current global phase and set the nested level to 1. 
- If the current thread was already in the critical section, we just need to increment the nested level.

`unlock`
^^^^^^^^

- We need to decrement the nested level. 
- If the nested level becomes 0, it means the thread is now in a quiescent state, we can notify the waiting writer thread if there is any.

`retire`
^^^^^^^^

- We need to push the retired callback to the `__retired_callback_queue_`

`synchronize`
^^^^^^^^^^^^^

- We need to go through two phases of grace period
- For each phase of the grace period, we need to flip the phase, and wait until all the reading threads are either

  - in a quiescent state, or
  - in the critical section with the new phase.

- After each phase ends, we can move the callbacks to the next phase's queue, and we can evaluate the callbacks that have gone through two phases.


Design Questions
================

What thread(s) should the deleters run on
-----------------------------------------

Paul E. McKenney suggested to run those deleters on a background thread:

  Folly currently uses an "inline executor", though some would like a
  separate thread in some cases.  The userspace RCU library always uses
  background threads.  By default only one, but there are ways to configure
  more. 

We have doubts on this. As a standard library, we usually don't spin threads for users. And Paul E. McKenney 
has pointed out that:

  Each library will have its own rules.
  One advantage of an inline executor is that the deleters apply
  backpressure against threads that flood the system with either .retire()
  or rcu_retire().  Some compensation for the additional deadlocks,
  I guess.  But you can get the same effect by properly restricting where
  the background threads run.

And Thomas Rodgers (libstdc++'s RCU implementer) also said:

  We don't normally do this, no, and I don't think I'd ever directly pursue such an approach in isolation. If, 
  on the other hand I could derive a permission from the user of library to create threads, by explicit use 
  of a system thread pool executor, perhaps then it would be ok to also use that executor to run other background
  threads for things like this. Having said that, I think that is something that SG1 should probably discuss. On the
  one hand spinning up a 'system' thread pool is going be potentially expensive and if the library user has done so,
  you probably would not release those resources until termination. So, in that sense, you've 'paid' for a background
  thread. But on the other hand, users of the library might find it surprising that then gave the library permission
  to also do work on those threads in ways they might not expect.

In libc++'s design, we would like to follow Folly's approach to run these deleters inline when `run_synchronize` or
`run_barrier` is called.

When should we run the deleters
-------------------------------

If we were to use the background thread approach, the deleters can be evaluated at any time after the readers have exited
their critical sections and objects are safe to reclaim. e.g. Drain the queue periodically. However, if we were to run the deleters
inline, we have to decide when to run them.

There are few places we can run the deleters:

- Inside `run_synchronize` after the writer thread is unblocked. This approach has the advantage that the writer thread
  can reclaim the retired objects as soon as possible. However, it has the disadvantage that the writer thread may be
  blocked for a long time if there are many retired objects to reclaim.

- Inside `run_barrier` after the writer thread is unblocked. Since `run_barrier` is designed to block until all the retired
  objects that happen before the `run_barrier` call are reclaimed, it is natural to run the deleters here.

- Inside `run_retire` after the deleter is put into the queue. `run_retire` is designed not to block. However, if there are
  objects that are safe to reclaim due to readers have exited their critical sections, and at the same time, there is no other
  writer threads that are currently draining the deleter queue, we can take the opportunity to run the deleters.

Folly currently takes the inline approach and runs the deleters at all three places mentioned above.

In libc++'s design, we would like to start with the inline approach and only run the deleters inside `run_synchronize` and
`run_barrier` for simplicity. Whether or not running the deleters inside `run_retire` is debatable. On the one hand, running
them will make `run_retire`  take more time to return. But not running them will make the retired objects stay in the queue
for a longer time. In an extreme case, if the user never calls `run_synchronize` or `run_barrier` after calling `run_retire`,
the retired objects will never be reclaimed (without a background thread to drain the queue).

Almost all APIs are `noexcept` . Is it designed to avoid memory allocation and avoid using `mutex` ?
-----------------------------------------------------------------------------------------------------

We should avoid using `mutex` as much as we can. However, both Folly and Paul's implementation use `mutex` in some places.

Paul E. McKenney mentioned that:

  On the non-noexcept nature of mutexes, the best I could do was to say
  that my reference implementation acquired its pthread_mutexes from C

There are few places that might need to use `mutex`:

- The first time a thread registers its existence with the `rcu` implementation. Depending on the implementation, this
  can happen inside the `rcu_*` calls.

- The add/remove of the deleter queue, if the queue is not a lock-free data structure.

- The destruction of a thread, which might need to do some clean up work for the `rcu` states.

Memory Allocation can throw as well. Luckily `run_retire` is not marked as `noexcept` in the paper, as `run_retire` is
likely to allocate memory to create a type erased object into the deleter queue.

However, there are other places where it might need to allocate memory. For example, see Folly's comments
`here <https://github.com/facebook/folly/blob/c83b50725f554c4912a73b676746242a22211eca/folly/synchronization/Rcu.h#L343>`__ ::

   * Note that despite the function being marked noexcept, an allocation
   * may take place in folly::ThreadLocal the first time a thread enters a read
   * region. Regardless, for now, we're marking this as noexcept to match the
   * N4895 standard proposal:
   *
   * https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/n4895.pdf
   */
  FOLLY_ALWAYS_INLINE void lock() noexcept {
    counters_.increment(version_.load(std::memory_order_acquire));
  }
  FOLLY_ALWAYS_INLINE void unlock() noexcept { counters_.decrement(); }

In libc++'s design, we will start with as less `mutex` usage as possible. In places where it is unavoidable, we could either
wrap the internal API with a while loop, or use the `atomic_unique_lock` that we introduced with `stop_token` implementations.
For memory allocations, we will only need them the first time we create some rcu_domain internal states or the first time registers a thread.
We will consider whether we accepts a `terminate` call if this fails to allocate.


How does `rcu` know about a thread?
------------------------------------

All design options require the `rcu_domain` to know whether a thread is in a critical section and stores each thread's state.
The question is how does the `rcu_domain` know about a thread?

Paul E. McKenney suggested that:

  In a library, I also suggest automating each thread's initial call to register_thread().

There are few options to automate the thread registration:

- In the `std::thread` and `std::jthread` constructors, we can register them to the `rcu_domain` 's state. However, this approach
  has the disadvantage that it does not cover threads that are not created by `std::thread` or `std::jthread` , for example, users
  can create a thread through `pthread_create` and use that thread to call the `rcu` APIs. Another disadvantage is that most users
  who create threads might not use `rcu` at all, so it might be wasteful to register all the threads by default.

- In the `rcu_domain::lock` , we can check whether the current thread is registered, if not, we can register it. This approach has
  the advantage that it can cover all threads, and it does not have the overhead of registering threads that do not use `rcu` at all.
  However, it has the disadvantage that it might have a performance impact on the first call to `rcu_domain::lock` for each thread.
  As the performance of `rcu_domain::lock` is critical, we need to make sure the thread registration process is as efficient as possible.

Folly uses the second approach to automate the thread registration. Another caveat of this approach in Folly is that, the first call to
`rcu_domain::lock` from a thread requires allocating memory to create the thread's state in the `rcu_domain` and it also needs to lock a
`mutex` to protect the thread registration process.

In libc++'s design, we will also use the second approach to automate the thread registration.

Is `rcu_domain` singleton?
--------------------------

The APIs take a reference to `rcu_domain` with default value::

    void rcu_synchronize(rcu_domain& dom = rcu_default_domain()) noexcept;

    void rcu_barrier(rcu_domain& dom = rcu_default_domain()) noexcept;

    template<class T, class D = default_delete<T>>
    void rcu_retire(T* p, D d = D(), rcu_domain& dom = rcu_default_domain());

However, the class `rcu_domain` has no user visible constructors::

    class rcu_domain {
      public:
        rcu_domain(const rcu_domain&) = delete;
        rcu_domain& operator=(const rcu_domain&) = delete;
    };

Paul E. McKenney confirmed that:

  Yes, right now, just the one instance.  Right now, its only purpose is to support C++ RAII RCU readers.
  Later, we might add the ability to create separate RCU instances, similar to SRCU in the Linux kernel.

Thomas Rodgers (libstdc++'s RCU implementer) also said:

  Currently, liburcu doesn't actually support discrete domains, so internal to the shared library there is
  only a single static instance of a type that wraps the relevant operations from the liburcu sources.
  The `uintptr_t`` member is initialized with the address of this instance, and its size. Future implementations
  could do something different behind this ABI contract, including a non-liburcu based approach. 

In libc++'s design, we will follow the same singleton pattern to simplify the design and implementation. Otherwise,
lots of thread local storages will require a per-instance state.

What about the ABI?
-------------------

The `rcu` library most functions are non-template. So it is possible to put the entire implementation details inside the dylib.

Existing Implementations
========================

The C++ paper `Read-Copy Update (RCU) <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2545r4.pdf>`__
discussed two reference implementations:

- Folly
- liburcu

Folly
-----

liburcu
-------

There are 4 flavors of `rcu` implementations in liburcu (see doc `here <https://github.com/urcu/userspace-rcu/blob/master/README.md#usage-of-all-urcu-libraries>`__):

- `memb`
- `qsbr`
- `mb`
- `bp`


libstdc++
---------

According to Thomas Rodgers (libstdc++'s RCU implementer), their plan is to embed the subset of `liburcu` that implements the `memb`` and `mb` RCU flavors, with `memb` being the preferred choice, 
based on the presence of `SYS_membarrier` . Because `liburcu` is license compatible with libstdc++ they can directly embed the relevant source from `liburcu`


