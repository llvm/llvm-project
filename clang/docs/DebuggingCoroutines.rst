========================
Debugging C++ Coroutines
========================

.. contents::
   :local:

Introduction
============

For performance and other architectural reasons, the C++ Coroutines feature in
the Clang compiler is implemented in two parts of the compiler.  Semantic
analysis is performed in Clang, and Coroutine construction and optimization
takes place in the LLVM middle-end.

However, this design forces us to generate insufficient debugging information.
Typically, the compiler generates debug information in the Clang frontend, as
debug information is highly language specific. However, this is not possible
for Coroutine frames because the frames are constructed in the LLVM middle-end.

To mitigate this problem, the LLVM middle end attempts to generate some debug
information, which is unfortunately incomplete, since much of the language
specific information is missing in the middle end.

This document describes how to use this debug information to better debug
coroutines.

Terminology
===========

Due to the recent nature of C++20 Coroutines, the terminology used to describe
the concepts of Coroutines is not settled.  This section defines a common,
understandable terminology to be used consistently throughout this document.

coroutine type
--------------

A `coroutine function` is any function that contains any of the Coroutine
Keywords `co_await`, `co_yield`, or `co_return`.  A `coroutine type` is a
possible return type of one of these `coroutine functions`.  `Task` and
`Generator` are commonly referred to coroutine types.

coroutine
---------

By technical definition, a `coroutine` is a suspendable function. However,
programmers typically use `coroutine` to refer to an individual instance.
For example:

.. code-block:: c++

  std::vector<Task> Coros; // Task is a coroutine type.
  for (int i = 0; i < 3; i++)
    Coros.push_back(CoroTask()); // CoroTask is a coroutine function, which
                                 // would return a coroutine type 'Task'.

In practice, we typically say "`Coros` contains 3 coroutines" in the above
example, though this is not strictly correct.  More technically, this should
say "`Coros` contains 3 coroutine instances" or "Coros contains 3 coroutine
objects."

In this document, we follow the common practice of using `coroutine` to refer
to an individual `coroutine instance`, since the terms `coroutine instance` and
`coroutine object` aren't sufficiently defined in this case.

coroutine frame
---------------

The C++ Standard uses `coroutine state` to describe the allocated storage. In
the compiler, we use `coroutine frame` to describe the generated data structure
that contains the necessary information.

The structure of coroutine frames
=================================

The structure of coroutine frames is defined as:

.. code-block:: c++

  struct {
    void (*__r)(); // function pointer to the `resume` function
    void (*__d)(); // function pointer to the `destroy` function
    promise_type; // the corresponding `promise_type`
    ... // Any other needed information
  }

In the debugger, the function's name is obtainable from the address of the
function. And the name of `resume` function is equal to the name of the
coroutine function. So the name of the coroutine is obtainable once the
address of the coroutine is known.

Print promise_type
==================

Every coroutine has a `promise_type`, which defines the behavior
for the corresponding coroutine. In other words, if two coroutines have the
same `promise_type`, they should behave in the same way.
To print a `promise_type` in a debugger when stopped at a breakpoint inside a
coroutine, printing the `promise_type` can be done by:

.. parsed-literal::

  print __promise

It is also possible to print the `promise_type` of a coroutine from the address
of the coroutine frame. For example, if the address of a coroutine frame is
0x416eb0, and the type of the `promise_type` is `task::promise_type`, printing
the `promise_type` can be done by:

.. parsed-literal::

  print (task::promise_type)*(0x416eb0+0x10)

This is possible because the `promise_type` is guaranteed by the ABI to be at a
16 bit offset from the coroutine frame.

Note that there is also an ABI independent method:

.. parsed-literal::

  print std::coroutine_handle<task::promise_type>::from_address((void*)0x416eb0).promise()

The functions `from_address(void*)` and `promise()` are often small enough to
be removed during optimization, so this method may not be possible.

Print coroutine frames
======================

LLVM generates the debug information for the coroutine frame in the LLVM middle
end, which permits printing of the coroutine frame in the debugger. Much like
the `promise_type`, when stopped at a breakpoint inside a coroutine we can
print the coroutine frame by:

.. parsed-literal::

  print __coro_frame


Just as printing the `promise_type` is possible from the coroutine address,
printing the details of the coroutine frame from an address is also possible:

.. parsed-literal::

  (gdb) # Get the address of coroutine frame
  (gdb) print/x *0x418eb0
  $1 = 0x4019e0
  (gdb) # Get the linkage name for the coroutine
  (gdb) x 0x4019e0
  0x4019e0 <_ZL9coro_taski>:  0xe5894855
  (gdb) # The coroutine frame type is 'linkage_name.coro_frame_ty'
  (gdb) print  (_ZL9coro_taski.coro_frame_ty)*(0x418eb0)
  $2 = {__resume_fn = 0x4019e0 <coro_task(int)>, __destroy_fn = 0x402000 <coro_task(int)>, __promise = {...}, ...}

The above is possible because:

(1) The name of the debug type of the coroutine frame is the `linkage_name`,
plus the `.coro_frame_ty` suffix because each coroutine function shares the
same coroutine type.

(2) The coroutine function name is accessible from the address of the coroutine
frame.

The above commands can be simplified by placing them in debug scripts.

Examples to print coroutine frames
----------------------------------

The print examples below use the following definition:

.. code-block:: c++

  #include <coroutine>
  #include <iostream>

  struct task{
    struct promise_type {
      task get_return_object() { return std::coroutine_handle<promise_type>::from_promise(*this); }
      std::suspend_always initial_suspend() { return {}; }
      std::suspend_always final_suspend() noexcept { return {}; }
      void return_void() noexcept {}
      void unhandled_exception() noexcept {}

      int count = 0;
    };

    void resume() noexcept {
      handle.resume();
    }

    task(std::coroutine_handle<promise_type> hdl) : handle(hdl) {}
    ~task() {
      if (handle)
        handle.destroy();
    }

    std::coroutine_handle<> handle;
  };

  class await_counter : public std::suspend_always {
    public:
      template<class PromiseType>
      void await_suspend(std::coroutine_handle<PromiseType> handle) noexcept {
          handle.promise().count++;
      }
  };

  static task coro_task(int v) {
    int a = v;
    co_await await_counter{};
    a++;
    std::cout << a << "\n";
    a++;
    std::cout << a << "\n";
    a++;
    std::cout << a << "\n";
    co_await await_counter{};
    a++;
    std::cout << a << "\n";
    a++;
    std::cout << a << "\n";
  }

  int main() {
    task t = coro_task(43);
    t.resume();
    t.resume();
    t.resume();
    return 0;
  }

In debug mode (`O0` + `g`), the printing result would be:

.. parsed-literal::

  {__resume_fn = 0x4019e0 <coro_task(int)>, __destroy_fn = 0x402000 <coro_task(int)>, __promise = {count = 1}, v = 43, a = 45, __coro_index = 1 '\001', struct_std__suspend_always_0 = {__int_8 = 0 '\000'},
    class_await_counter_1 = {__int_8 = 0 '\000'}, class_await_counter_2 = {__int_8 = 0 '\000'}, struct_std__suspend_always_3 = {__int_8 = 0 '\000'}}

In the above, the values of `v` and `a` are clearly expressed, as are the
temporary values for `await_counter` (`class_await_counter_1` and
`class_await_counter_2`) and `std::suspend_always` (
`struct_std__suspend_always_0` and `struct_std__suspend_always_3`). The index
of the current suspension point of the coroutine is emitted as `__coro_index`.
In the above example, the `__coro_index` value of `1` means the coroutine
stopped at the second suspend point (Note that `__coro_index` is zero indexed)
which is the first `co_await await_counter{};` in `coro_task`. Note that the
first initial suspend point is the compiler generated
`co_await promise_type::initial_suspend()`.

However, when optimizations are enabled, the printed result changes drastically:

.. parsed-literal::

  {__resume_fn = 0x401280 <coro_task(int)>, __destroy_fn = 0x401390 <coro_task(int)>, __promise = {count = 1}, __int_32_0 = 43, __coro_index = 1 '\001'}

Unused values are optimized out, as well as the name of the local variable `a`.
The only information remained is the value of a 32 bit integer. In this simple
case, it seems to be pretty clear that `__int_32_0` represents `a`. However, it
is not true.

An important note with optimization is that the value of a variable may not
properly express the intended value in the source code.  For example:

.. code-block:: c++

  static task coro_task(int v) {
    int a = v;
    co_await await_counter{};
    a++; // __int_32_0 is 43 here
    std::cout << a << "\n";
    a++; // __int_32_0 is still 43 here
    std::cout << a << "\n";
    a++; // __int_32_0 is still 43 here!
    std::cout << a << "\n";
    co_await await_counter{};
    a++; // __int_32_0 is still 43 here!!
    std::cout << a << "\n";
    a++; // Why is __int_32_0 still 43 here?
    std::cout << a << "\n";
  }

When debugging step-by-step, the value of `__int_32_0` seemingly does not
change, despite being frequently incremented, and instead is always `43`.
While this might be surprising, this is a result of the optimizer recognizing
that it can eliminate most of the load/store operations. The above code gets
optimized to the equivalent of:

.. code-block:: c++

  static task coro_task(int v) {
    store v to __int_32_0 in the frame
    co_await await_counter{};
    a = load __int_32_0
    std::cout << a+1 << "\n";
    std::cout << a+2 << "\n";
    std::cout << a+3 << "\n";
    co_await await_counter{};
    a = load __int_32_0
    std::cout << a+4 << "\n";
    std::cout << a+5 << "\n";
  }

It should now be obvious why the value of `__int_32_0` remains unchanged
throughout the function. It is important to recognize that `__int_32_0`
does not directly correspond to `a`, but is instead a variable generated
to assist the compiler in code generation. The variables in an optimized
coroutine frame should not be thought of as directly representing the
variables in the C++ source.

Get the suspended points
========================

An important requirement for debugging coroutines is to understand suspended
points, which are where the coroutine is currently suspended and awaiting.

For simple cases like the above, inspecting the value of the `__coro_index`
variable in the coroutine frame works well.

However, it is not quite so simple in really complex situations. In these
cases, it is necessary to use the coroutine libraries to insert the
line-number.

For example:

.. code-block:: c++

  // For all the promise_type we want:
  class promise_type {
    ...
  +  unsigned line_number = 0xffffffff;
  };

  #include <source_location>

  // For all the awaiter types we need:
  class awaiter {
    ...
    template <typename Promise>
    void await_suspend(std::coroutine_handle<Promise> handle,
                       std::source_location sl = std::source_location::current()) {
          ...
          handle.promise().line_number = sl.line();
    }
  };

In this case, we use `std::source_location` to store the line number of the
await inside the `promise_type`.  Since we can locate the coroutine function
from the address of the coroutine, we can identify suspended points this way
as well.

The downside here is that this comes at the price of additional runtime cost.
This is consistent with the C++ philosophy of "Pay for what you use".

Get the asynchronous stack
==========================

Another important requirement to debug a coroutine is to print the asynchronous
stack to identify the asynchronous caller of the coroutine.  As many
implementations of coroutine types store `std::coroutine_handle<> continuation`
in the promise type, identifying the caller should be trivial.  The
`continuation` is typically the awaiting coroutine for the current coroutine.
That is, the asynchronous parent.

Since the `promise_type` is obtainable from the address of a coroutine and
contains the corresponding continuation (which itself is a coroutine with a
`promise_type`), it should be trivial to print the entire asynchronous stack.

This logic should be quite easily captured in a debugger script.

Get the living coroutines
=========================

Another useful task when debugging coroutines is to enumerate the list of
living coroutines, which is often done with threads.  While technically
possible, this task is not recommended in production code as it is costly at
runtime. One such solution is to store the list of currently running coroutines
in a collection:

.. code-block:: c++

  inline std::unordered_set<void*> lived_coroutines;
  // For all promise_type we want to record
  class promise_type {
  public:
      promise_type() {
          // Note to avoid data races
          lived_coroutines.insert(std::coroutine_handle<promise_type>::from_promise(*this).address());
      }
      ~promise_type() {
          // Note to avoid data races
          lived_coroutines.erase(std::coroutine_handle<promise_type>::from_promise(*this).address());
      }
  };

In the above code snippet, we save the address of every lived coroutine in the
`lived_coroutines` `unordered_set`. As before, once we know the address of the
coroutine we can derive the function, `promise_type`, and other members of the
frame. Thus, we could print the list of lived coroutines from that collection.

Please note that the above is expensive from a storage perspective, and requires
some level of locking (not pictured) on the collection to prevent data races.
