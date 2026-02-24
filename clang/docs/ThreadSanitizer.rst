ThreadSanitizer
===============

Introduction
------------

ThreadSanitizer is a tool that detects data races.  It consists of a compiler
instrumentation module and a run-time library.  Typical slowdown introduced by
ThreadSanitizer is about **5x-15x**.  Typical memory overhead introduced by
ThreadSanitizer is about **5x-10x**.

How to build
------------

Build LLVM/Clang with `CMake <https://llvm.org/docs/CMake.html>`_.

Supported Platforms
-------------------

ThreadSanitizer is supported on the following OS:

* Android aarch64, x86_64
* Darwin arm64, x86_64
* FreeBSD
* Linux aarch64, x86_64, powerpc64, powerpc64le
* NetBSD

Support for other 64-bit architectures is possible, contributions are welcome.
Support for 32-bit platforms is problematic and is not planned.

Usage
-----

Simply compile and link your program with ``-fsanitize=thread``.  To get a
reasonable performance add ``-O1`` or higher.  Use ``-g`` to get file names
and line numbers in the warning messages.

Example:

.. code-block:: console

  % cat projects/compiler-rt/lib/tsan/lit_tests/tiny_race.c
  #include <pthread.h>
  int Global;
  void *Thread1(void *x) {
    Global = 42;
    return x;
  }
  int main() {
    pthread_t t;
    pthread_create(&t, NULL, Thread1, NULL);
    Global = 43;
    pthread_join(t, NULL);
    return Global;
  }

  $ clang -fsanitize=thread -g -O1 tiny_race.c

If a bug is detected, the program will print an error message to stderr.
Currently, ThreadSanitizer symbolizes its output using an external
``addr2line`` process (this will be fixed in future).

.. code-block:: bash

  % ./a.out
  WARNING: ThreadSanitizer: data race (pid=19219)
    Write of size 4 at 0x7fcf47b21bc0 by thread T1:
      #0 Thread1 tiny_race.c:4 (exe+0x00000000a360)

    Previous write of size 4 at 0x7fcf47b21bc0 by main thread:
      #0 main tiny_race.c:10 (exe+0x00000000a3b4)

    Thread T1 (running) created at:
      #0 pthread_create tsan_interceptors.cc:705 (exe+0x00000000c790)
      #1 main tiny_race.c:9 (exe+0x00000000a3a4)

``__has_feature(thread_sanitizer)``
------------------------------------

In some cases one may need to execute different code depending on whether
ThreadSanitizer is enabled.
:ref:`\_\_has\_feature <langext-__has_feature-__has_extension>` can be used for
this purpose.

.. code-block:: c

    #if defined(__has_feature)
    #  if __has_feature(thread_sanitizer)
    // code that builds only under ThreadSanitizer
    #  endif
    #endif

``__attribute__((no_sanitize("thread")))``
-----------------------------------------------

Some code should not be instrumented by ThreadSanitizer.  One may use the
function attribute ``no_sanitize("thread")`` to disable instrumentation of plain
(non-atomic) loads/stores in a particular function.  ThreadSanitizer still
instruments such functions to avoid false positives and provide meaningful stack
traces.  This attribute may not be supported by other compilers, so we suggest
to use it together with ``__has_feature(thread_sanitizer)``.

``__attribute__((disable_sanitizer_instrumentation))``
--------------------------------------------------------

The ``disable_sanitizer_instrumentation`` attribute can be applied to functions
to prevent all kinds of instrumentation. As a result, it may introduce false
positives and incorrect stack traces. Therefore, it should be used with care,
and only if absolutely required; for example for certain code that cannot
tolerate any instrumentation and resulting side-effects. This attribute
overrides ``no_sanitize("thread")``.

Interaction of Inlining with Disabling Sanitizer Instrumentation
-----------------------------------------------------------------

* A `no_sanitize` function will not be inlined heuristically by the compiler into a sanitized function.
* An `always_inline` function will adopt the instrumentation status of the function it is inlined into.
* Forcibly combining `no_sanitize` and ``__attribute__((always_inline))`` is not supported, and will often lead to unexpected results. To avoid mixing these attributes, use:

. code-block:: c

    // Note, __has_feature test for sanitizers is deprecated, and Clang will support __SANITIZE_<sanitizer>__ similar to GCC.
    #if __has_feature(thread_sanitizer) || defined(__SANITIZE_THREAD__) || ... <other sanitizers>
    #define ALWAYS_INLINE_IF_UNINSTRUMENTED
    #else
    #define ALWAYS_INLINE_IF_UNINSTRUMENTED __attribute__((always_inline))
    #endif

Explicit Sanitizer Checks with ``__builtin_allow_sanitize_check``
-----------------------------------------------------------------

The ``__builtin_allow_sanitize_check("thread")`` builtin can be used to
conditionally execute code depending on whether ThreadSanitizer checks are
enabled and permitted by the current policy (after inlining). This is
particularly useful for inserting explicit, sanitizer-specific checks around
operations like syscalls or inline assembly, which might otherwise be unchecked
by the sanitizer.

Example:

.. code-block:: c

    void __tsan_read8(void *);

    inline __attribute__((always_inline))
    void my_helper(void *addr) {
      if (__builtin_allow_sanitize_check("thread"))
        __tsan_read8(addr);
      // ... actual logic, e.g. inline assembly ...
      asm volatile ("..." : : "r" (addr) : "memory");
    }

    void instrumented_function() {
      ...
      my_helper(&shared_data); // checks are active
      ...
    }

    __attribute__((no_sanitize("thread")))
    void uninstrumented_function() {
      ...
      my_helper(&shared_data); // checks are skipped
      ...
    }

Ignorelist
----------

ThreadSanitizer supports ``src`` and ``fun`` entity types in
:doc:`SanitizerSpecialCaseList`, that can be used to suppress data race reports
in the specified source files or functions. Unlike functions marked with
``no_sanitize("thread")`` attribute, ignored functions are not instrumented
at all. This can lead to false positives due to missed synchronization via
atomic operations and missed stack frames in reports.

Limitations
-----------

* ThreadSanitizer uses more real memory than a native run. At the default
  settings the memory overhead is 5x plus 1Mb per each thread. Settings with 3x
  (less accurate analysis) and 9x (more accurate analysis) overhead are also
  available.
* ThreadSanitizer maps (but does not reserve) a lot of virtual address space.
  This means that tools like ``ulimit`` may not work as usually expected.
* Libc/libstdc++ static linking is not supported.
* Non-position-independent executables are not supported.  Therefore, the
  ``fsanitize=thread`` flag will cause Clang to act as though the ``-fPIE``
  flag had been supplied if compiling without ``-fPIC``, and as though the
  ``-pie`` flag had been supplied if linking an executable.

Security Considerations
-----------------------

ThreadSanitizer is a bug detection tool and its runtime is not meant to be
linked against production executables. While it may be useful for testing,
ThreadSanitizer's runtime was not developed with security-sensitive
constraints in mind and may compromise the security of the resulting executable.

Current Status
--------------

ThreadSanitizer is in beta stage.  It is known to work on large C++ programs
using pthreads, but we do not promise anything (yet).  C++11 threading is
supported with llvm libc++.  The test suite is integrated into CMake build
and can be run with ``make check-tsan`` command.

We are actively working on enhancing the tool --- stay tuned.  Any help,
especially in the form of minimized standalone tests is more than welcome.

Adaptive Delay
--------------

Overview
~~~~~~~~

Adaptive Delay is an optional ThreadSanitizer feature that injects delays at
synchronization points to explore novel thread interleavings and increase the
likelihood of exposing data races. By perturbing thread scheduling, adaptive
delay creates more opportunities for concurrent accesses to shared data,
improving race detection.

Adaptive delay is particularly useful for:

* Detecting races in rarely-executed thread interleavings or code paths
* Testing parallel data structures and algorithms

When enabled, adaptive delay maintains a configurable time budget to balance
race exposure against performance overhead. The delays can be

 * random amount of spin cycles
 * a single yield to the OS scheduler
 * random usleep

The strategy prioritizes high-value synchronization points:

* Relaxed atomic operations receive cheap delays (spin cycles) with low sampling
* Synchronizing atomic operations (acquire/release/seq_cst) receive moderate
  delays with higher sampling
* Mutex and thread lifecycle operations receive the longest delays with highest
  sampling

The delays focus on synchronization points with clear happens-before relationships,
as those are most likely to expose data races.

Enabling Adaptive Delay
~~~~~~~~~~~~~~~~~~~~~~~

Adaptive delay is disabled by default. Enable it by setting the
``enable_adaptive_delay`` flag:

.. code-block:: console

  $ TSAN_OPTIONS=enable_adaptive_delay=1 ./myapp

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Adaptive Delay Options
   :name: adaptive-delay-options-table
   :header-rows: 1
   :widths: 35 10 15 40

   * - Flag
     - Type
     - Default
     - Description
   * - ``enable_adaptive_delay``
     - bool
     - false
     - Enable adaptive delay injection to expose data races.
   * - ``adaptive_delay_aggressiveness``
     - int
     - 25
     - Controls delay injection intensity for race detection. Higher values inject
       more delays to expose races. Value must be greater than 0. Suggested values:
       10 (minimal), 50 (moderate), 200 (aggressive). This is a tuning parameter;
       actual overhead varies by workload and platform.
   * - ``adaptive_delay_relaxed_sample_rate``
     - int
     - 10000
     - Sample 1 in N relaxed atomic operations for delay injection. Relaxed atomics
       have minimal synchronization, so sampling helps avoid excessive overhead.
   * - ``adaptive_delay_sync_atomic_sample_rate``
     - int
     - 100
     - Sample 1 in N acquire/release/seq_cst atomic operations for delay injection.
       These synchronizing atomics are more likely to expose races, so are sampled
       more often.
   * - ``adaptive_delay_mutex_sample_rate``
     - int
     - 10
     - Sample 1 in N mutex/condition variable operations for delay injection. Mutex
       ops are high-value synchronization points and are sampled frequently.
   * - ``adaptive_delay_max_atomic``
     - string
     - ``"sleep_us=50"``
     - Maximum delay for atomic operations. Format: ``"spin=N"`` (N spin cycles,
       1 <= N <= 10,000), ``"yield"`` (one yield to the OS), or ``"sleep_us=N"``
       (up to N microseconds). The delay is randomly chosen up to the specified
       maximum N.
   * - ``adaptive_delay_max_sync``
     - string
     - ``"sleep_us=500"``
     - Maximum delay for synchronization operations (mutex and thread lifecycle
       operations). Format: same as ``adaptive_delay_max_atomic``. Typically set
       longer than atomic delays since these operations involve waking blocked threads
       and may be more likely to expose races.

Examples
~~~~~~~~

Enable adaptive delay with moderate aggressiveness:

.. code-block:: console

  $ TSAN_OPTIONS=enable_adaptive_delay=1:adaptive_delay_aggressiveness=50 ./myapp

Enable aggressive delay injection:

.. code-block:: console

  $ TSAN_OPTIONS=enable_adaptive_delay=1:adaptive_delay_aggressiveness=200 ./myapp

Increase sampling frequency for mutex operations:

.. code-block:: console

  $ TSAN_OPTIONS=enable_adaptive_delay=1:adaptive_delay_mutex_sample_rate=5 ./myapp

Simulation Scheduler
--------------------

Overview
~~~~~~~~

The Simulation Scheduler is an optional ThreadSanitizer feature that enables
systematic exploration of thread interleavings to expose data races that may be
difficult to trigger in normal execution. Unlike standard ThreadSanitizer which
detects races as they occur naturally during program execution, the simulation
scheduler takes control of thread scheduling to deliberately explore different
execution orderings.

Simulation is particularly useful for:

* Testing concurrent data structure or algorithms during development to ensure
  correctness (for example, a lock free queue).
* Finding races in rarely-executed interleavings that standard TSAN may miss
* Reproducing specific race conditions deterministically

Simulation is not useful for running full applications, and will likely not
work in these scenarios. The code run in simulation should almost always be a
small unit test exercising very specific functionality.

When enabled via the ``__tsan_simulate()`` API, the simulation scheduler runs
the program's concurrent code multiple times (iterations), choosing different
thread interleavings in each iteration. The scheduler injects context switches
at synchronization points (atomic operations, mutex operations, thread lifecycle
events) to maximize coverage of possible execution orderings. If a data race is
detected, the simulation stops and reports which iteration exposed the race,
allowing that specific interleaving to be reproduced.

Usage
~~~~~

To use simulation, wrap the concurrent code you want to test in a callback
function and invoke it through the ``__tsan_simulate()`` API:

.. code-block:: c

    extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

    void test_concurrent_code(void *arg) {
      // Create threads, run concurrent operations
      pthread_t t1, t2;
      pthread_create(&t1, NULL, thread_func, NULL);
      pthread_create(&t2, NULL, thread_func, NULL);
      pthread_join(t1, NULL);
      pthread_join(t2, NULL);
    }

    int main() {
      return __tsan_simulate(test_concurrent_code, NULL);
    }

Then compile with ThreadSanitizer and enable the simulation scheduler:

.. code-block:: console

  $ clang -fsanitize=thread -g -O1 mytest.c
  $ TSAN_OPTIONS=simulate_scheduler=random ./a.out
  ThreadSanitizer: simulation starting (iterations 0..999, max_depth=10000, scheduler=random)

Automatic Main Wrapping
~~~~~~~~~~~~~~~~~~~~~~~~

For convenience, the ``-fsanitize-thread-simulate-main`` compiler flag
automatically wraps ``main()`` to call ``__tsan_simulate()``, eliminating the
need to manually modify code:

.. code-block:: c

    // No need to call __tsan_simulate() manually
    void *thread_func(void *arg) { /* ... */ }

    int main() {
      // This entire main() runs under simulation automatically
      pthread_t t1, t2;
      pthread_create(&t1, NULL, thread_func, NULL);
      pthread_create(&t2, NULL, thread_func, NULL);
      pthread_join(t1, NULL);
      pthread_join(t2, NULL);
      return 0;
    }

Compile and run:

.. code-block:: console

  $ clang -fsanitize=thread -fsanitize-thread-simulate-main -g -O1 mytest.c
  $ TSAN_OPTIONS=simulate_scheduler=random ./a.out
  ThreadSanitizer: simulation starting (iterations 0..999, max_depth=10000, scheduler=random)

**Platform Support**: This flag requires GNU ld linker support for ``--wrap=main``
and is currently only supported on Linux. Do not manually specify ``-Wl,--wrap=main``
when using this flag, as the compiler handles the wrapping automatically.

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Simulation Scheduler Options
   :name: simulation-scheduler-options-table
   :header-rows: 1
   :widths: 35 10 15 40

   * - Flag
     - Type
     - Default
     - Description
   * - ``simulate_scheduler``
     - string
     - ""
     - Scheduler algorithm for simulation. Supported values: ``"random"`` for
       random scheduling decisions. Empty string (default) means simulation is
       disabled. Must be set to enable simulation.
   * - ``simulate_iterations``
     - int
     - 1000
     - Number of iterations to run. Each iteration explores a different thread
       interleaving. More iterations increase the likelihood of finding races but
       take longer to complete.
   * - ``simulate_start_iteration``
     - int
     - 0
     - Starting iteration number. Useful for reproducing specific iteration
       failures. Set this to the iteration number reported when a race was found
       to reproduce that exact interleaving.
   * - ``simulate_max_depth``
     - int
     - 10000
     - Maximum number of scheduling decisions per iteration. If exceeded, the
       iteration is aborted and simulation returns an error. Prevents infinite
       loops or excessive scheduling overhead.
   * - ``simulate_schedule_probability``
     - int
     - 100
     - Probability (0-100%) of performing a context switch at each scheduling
       point. Lower values (e.g., 0) disable context switching, allowing threads
       to run more sequentially. Useful for comparing simulation results against
       sequential execution.
   * - ``simulate_schedule_on_memory_access``
     - bool
     - false
     - Insert scheduling points at every memory read/write during simulation for
       maximum interleaving exploration. This can significantly increase overhead
       but may expose additional races.
   * - ``simulate_print_schedule_stacks``
     - bool
     - false
     - Print stack trace at each scheduling point. Useful for debugging and
       understanding exact interleavings, but generates significant output.

Examples
~~~~~~~~

Basic race detection that standard TSAN rarely finds:

.. code-block:: c

    // Compile: clang -fsanitize=thread -g -O1 test.c
    #include <pthread.h>
    #include <stdatomic.h>

    extern int __tsan_simulate(void (*callback)(void *), void *arg);

    atomic_int d = 0;
    int a = 0;  // Non-atomic - race target

    void *thread_func(void *arg) {
      atomic_fetch_add(&d, 1);
      ++a;  // Data race!
      atomic_fetch_add(&d, 1);
      return NULL;
    }

    void test_callback(void *arg) {
      pthread_t t1, t2;
      pthread_create(&t1, NULL, thread_func, NULL);
      pthread_create(&t2, NULL, thread_func, NULL);
      pthread_join(t1, NULL);
      pthread_join(t2, NULL);
    }

    int main() { return __tsan_simulate(test_callback, NULL); }

Standard TSAN execution rarely detects this race. Running 100 times produces no
output most of the time:

.. code-block:: console

  $ clang -fsanitize=thread -g -O1 test.c
  $ for i in {1..100}; do ./a.out; done
  (no output - race not detected)

Run with simulation enabled:

.. code-block:: console

  $ TSAN_OPTIONS=simulate_scheduler=random:simulate_iterations=50 ./a.out
  ThreadSanitizer: simulation starting (iterations 0..999, max_depth=10000, scheduler=random)
  ==================
  WARNING: ThreadSanitizer: data race
    Write of size 4 at 0x... by thread T1:
      #0 thread_func test.c:12

    Previous write of size 4 at 0x... by thread T2:
      #0 thread_func test.c:12
  ==================
  ThreadSanitizer: data race detected at iteration 4
  ThreadSanitizer: to reproduce, set TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=4
  ThreadSanitizer: simulation stopped due to race detection after 5 iterations

To reproduce the specific iteration that found the race:

.. code-block:: console

  $ TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=4:simulate_iterations=1 ./a.out
  ThreadSanitizer: simulation starting (iterations 4..4, max_depth=10000, scheduler=random)
  ==================
  WARNING: ThreadSanitizer: data race
  ...

Compare simulation results with sequential execution (no context switching):

.. code-block:: console

  $ TSAN_OPTIONS=simulate_scheduler=random:simulate_schedule_probability=0:simulate_iterations=100 ./a.out

Deadlock detection
~~~~~~~~~~~~~~~~~~

Simulation detects when an actual deadlock occurs, i.e., no thread is runnable and the program
will remain blocked forever. For example,

.. code-block:: c

    // Compile: clang -fsanitize=thread -g -O1 deadlock.c
    #include <pthread.h>

    extern int __tsan_simulate(void (*callback)(void *), void *arg);

    pthread_mutex_t mutex;
    pthread_cond_t condvar;

    void *thread_func(void *arg) {
      pthread_mutex_lock(&mutex);
      // Wait on condition variable that will never be signaled
      pthread_cond_wait(&condvar, &mutex);
      pthread_mutex_unlock(&mutex);
      return NULL;
    }

    void test_callback(void *arg) {
      pthread_mutex_init(&mutex, NULL);
      pthread_cond_init(&condvar, NULL);

      pthread_t t1;
      pthread_create(&t1, NULL, thread_func, NULL);
      pthread_join(t1, NULL);

      pthread_cond_destroy(&condvar);
      pthread_mutex_destroy(&mutex);
    }

    int main() { return __tsan_simulate(test_callback, NULL); }

Run with simulation:

.. code-block:: console

  $ TSAN_OPTIONS=simulate_scheduler=random:simulate_iterations=2 ./deadlock
  ThreadSanitizer: simulation starting (iterations 0..1, max_depth=10000, scheduler=random)
  ThreadSanitizer: deadlock detected at iteration 0 - all threads are blocked
  ThreadSanitizer: to reproduce, set TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=0

More Information
----------------
`<https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual>`_
