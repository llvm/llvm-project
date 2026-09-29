# Benchmarking `llvm-libc`'s memory functions

## Foreword

Microbenchmarks are valuable tools to assess and compare the performance of
isolated pieces of code. However they don't capture all interactions of complex
systems; and so other metrics can be equally important:

-   **code size** (to reduce instruction cache pressure),
-   **Profile Guided Optimization** friendliness,
-   **hyperthreading / multithreading** friendliness.

## Rationale

The goal here is to satisfy the [Benchmarking
Principles](https://en.wikipedia.org/wiki/Benchmark_(computing)#Benchmarking_Principles).

1.  **Relevance**: Benchmarks should measure relatively vital features.
2.  **Representativeness**: Benchmark performance metrics should be broadly
    accepted by industry and academia.
3.  **Equity**: All systems should be fairly compared.
4.  **Repeatability**: Benchmark results can be verified.
5.  **Cost-effectiveness**: Benchmark tests are economical.
6.  **Scalability**: Benchmark tests should measure from single server to
    multiple servers.
7.  **Transparency**: Benchmark metrics should be easy to understand.

Benchmarking is a [subtle
art](https://en.wikipedia.org/wiki/Benchmark_(computing)#Challenges) and
benchmarking memory functions is no exception. Here we'll dive into
peculiarities of designing good microbenchmarks for `llvm-libc` memory
functions.

## Challenges

As seen in the [README.md](README.md#stochastic-mode) the microbenchmarking
facility should focus on measuring **low latency code**. If copying a few bytes
takes in the order of a few cycles, the benchmark should be able to **measure
accurately down to the cycle**.

### Measuring instruments

There are different sources of time in a computer (ordered from high to low resolution)
 - [Performance
   Counters](https://en.wikipedia.org/wiki/Hardware_performance_counter): used to
   introspect the internals of the CPU,
 - [High Precision Event
   Timer](https://en.wikipedia.org/wiki/High_Precision_Event_Timer): used to
   trigger short lived actions,
 - [Real-Time Clocks (RTC)](https://en.wikipedia.org/wiki/Real-time_clock): used
   to keep track of the computer's time.

In theory **Performance Counters** provide cycle accurate measurement via the
`cpu cycles` event. But as we'll see, they are not really practical in this
context.

### Performance counters and modern processor architecture

Modern CPUs are [out of
order](https://en.wikipedia.org/wiki/Out-of-order_execution) and
[superscalar](https://en.wikipedia.org/wiki/Superscalar_processor) as a
consequence it is [hard to know what is included when the counter is
read](https://en.wikipedia.org/wiki/Hardware_performance_counter#Instruction_based_sampling),
some instructions may still be **in flight**, some others may be executing
[**speculatively**](https://en.wikipedia.org/wiki/Speculative_execution). As a
matter of fact **on the same machine, measuring twice the same piece of code will yield
different results.**

### Performance counters semantics inconsistencies and availability

Although they have the same name, the exact semantics of performance counters
are micro-architecture dependent: **it is generally not possible to compare two
micro-architectures exposing the same performance counters.**

Each vendor decides which performance counters to implement and their exact
meaning. Although we want to benchmark `llvm-libc` memory functions for all
available [target
triples](https://clang.llvm.org/docs/CrossCompilation.html#target-triple), there
are **no guarantees that the counter we're interested in is available.**

### Additional imprecisions

-   Reading performance counters is done through Kernel [System
    calls](https://en.wikipedia.org/wiki/System_call). The System call itself
    is costly (hundreds of cycles) and will perturbate the counter's value.
-   [Interruptions](https://en.wikipedia.org/wiki/Interrupt#Processor_response)
    can occur during measurement.
-   If the system is already under monitoring (virtual machines or system wide
    profiling) the kernel can decide to multiplex the performance counters
    leading to lower precision or even completely missing the measurement.
-   The Kernel can decide to [migrate the
    process](https://en.wikipedia.org/wiki/Process_migration) to a different
    core.
-   [Dynamic frequency
    scaling](https://en.wikipedia.org/wiki/Dynamic_frequency_scaling) can kick
    in during the measurement and change the ticking duration. **Ultimately we
    care about the amount of work over a period of time**. This removes some
    legitimacy of measuring cycles rather than **raw time**.

### Cycle accuracy conclusion

We have seen that performance counters are: not widely available, semantically
inconsistent across micro-architectures and imprecise on modern CPUs for small
snippets of code.

## Design decisions

In order to achieve the needed precision we would need to resort on more widely
available counters and derive the time from a high number of runs: going from a
single deterministic measure to a probabilistic one.

**To get a good signal to noise ratio we need the running time of the piece of
code to be orders of magnitude greater than the measurement precision.**

For instance, if measurement precision is of 10 cycles, we need the function
runtime to take more than 1000 cycles to achieve 1%
[SNR](https://en.wikipedia.org/wiki/Signal-to-noise_ratio).

### Repeating code N-times until precision is sufficient

The algorithm is as follows:

-   We measure the time it takes to run the code _N_ times (Initially _N_ is 10
    for instance)
-   We deduce an approximation of the runtime of one iteration (= _runtime_ /
    _N_).
-   We increase _N_ by _X%_ and repeat the measurement (geometric progression).
-   We keep track of the _one iteration runtime approximation_ and build a
    weighted mean of all the samples so far (weight is proportional to _N_)
-   We stop the process when the difference between the weighted mean and the
    last estimation is smaller than _ε_ or when other stopping conditions are
    met (total runtime, maximum iterations or maximum sample count).

This method allows us to be as precise as needed provided that the measured
runtime is proportional to _N_. Longer run times also smooth out imprecision
related to _interrupts_ and _context switches_.

Note: When measuring longer runtimes (e.g. copying several megabytes of data)
the above assumption doesn't hold anymore and the _ε_ precision cannot be
reached by increasing iterations. The whole benchmarking process becomes
prohibitively slow. In this case the algorithm is limited to a single sample and
repeated several times to get a decent 95% confidence interval.

### Effect of branch prediction

When measuring code with branches, repeating the same call again and again will
allow the processor to learn the branching patterns and perfectly predict all
the branches, leading to unrealistic results.

**Decision: When benchmarking small buffer sizes, the function parameters should
be randomized between calls to prevent perfect branch predictions.**

### Effect of the memory subsystem

The CPU is tightly coupled to the memory subsystem. It is common to see `L1`,
`L2` and `L3` data caches.

We may be tempted to randomize data accesses widely to exercise all the caching
layers down to RAM but the [cost of accessing lower layers of
memory](https://people.eecs.berkeley.edu/~rcs/research/interactive_latency.html)
completely dominates the runtime for small sizes.

So to respect **Equity** and **Repeatability** principles we should make sure we
**do not** depend on the memory subsystem.

**Decision: When benchmarking small buffer sizes, the data accessed by the
function should stay in `L1`.**

### Effect of prefetching

In case of small buffer sizes,
[prefetching](https://en.wikipedia.org/wiki/Cache_prefetching) should not kick
in but in case of large buffers it may introduce a bias.

**Decision: When benchmarking large buffer sizes, the data should be accessed in
a random fashion to lower the impact of prefetching between calls.**

### Effect of dynamic frequency scaling

Modern processors implement [dynamic frequency
scaling](https://en.wikipedia.org/wiki/Dynamic_frequency_scaling). In so-called
`performance` mode the CPU will increase its frequency and run faster than usual
within [some limits](https://en.wikipedia.org/wiki/Intel_Turbo_Boost) : _"The
increased clock rate is limited by the processor's power, current, and thermal
limits, the number of cores currently in use, and the maximum frequency of the
active cores."_

**Decision: When benchmarking we want to make sure the dynamic frequency scaling
is always set to `performance`. We also want to make sure that the time based
events are not impacted by frequency scaling.**

See [README.md](README.md) on how to set this up.

### Reserved and pinned cores

Some operating systems allow [core
reservation](https://stackoverflow.com/questions/13583146/whole-one-core-dedicated-to-single-process).
It removes a set of perturbation sources like: process migration, context
switches and interrupts. When a core is hyperthreaded, both cores should be
reserved.

## Microbenchmarks limitations

As stated in the Foreword section a number of effects do play a role in
production but are not directly measurable through microbenchmarks. The code
size of the benchmark is (much) smaller than the hot code of real applications
and **doesn't exhibit instruction cache pressure as much**.

### iCache pressure

Fundamental functions that are called frequently will occupy the L1 iCache
([illustration](https://en.wikipedia.org/wiki/CPU_cache#Example:_the_K8)). If
they are too big they will prevent other hot code to stay in the cache and incur
[stalls](https://en.wikipedia.org/wiki/CPU_cache#CPU_stalls). So the memory
functions should be as small as possible.

### iTLB pressure

The same reasoning goes for instruction Translation Lookaside Buffer
([iTLB](https://en.wikipedia.org/wiki/Translation_lookaside_buffer)) incurring
[TLB
misses](https://en.wikipedia.org/wiki/Translation_lookaside_buffer#TLB-miss_handling).

## FAQ

1.  Why don't you use Google Benchmark directly?

    We reuse some parts of Google Benchmark (detection of frequency scaling, CPU
    cache hierarchy informations) but when it comes to measuring memory
    functions Google Benchmark have a few issues:

    -   Google Benchmark privileges code based configuration via macros and
        builders. It is typically done in a static manner. In our case the
        parameters we need to setup are a mix of what's usually controlled by
        the framework (number of trials, maximum number of iterations, size
        ranges) and parameters that are more tied to the function under test
        (randomization strategies, custom values). Achieving this with Google
        Benchmark is cumbersome as it involves templated benchmarks and
        duplicated code. In the end, the configuration would be spread across
        command line flags (via framework's option or custom flags), and code
        constants.
    -   Output of the measurements is done through a `BenchmarkReporter` class,
        that makes it hard to access the parameters discussed above.
