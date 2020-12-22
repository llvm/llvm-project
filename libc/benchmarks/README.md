# Libc mem* benchmarks

This framework has been designed to evaluate and compare relative performance of memory function implementations on a particular machine.

It relies on two tools:
 - `libc-benchmark-main` a C++ benchmarking utility producing raw measurements,
 - `libc-benchmark-analysis.py3` a tool to process the measurements into reports.

## Benchmarking tool

### Setup

```shell
cd llvm-project
cmake -B/tmp/build -Sllvm -DLLVM_ENABLE_PROJECTS='clang;clang-tools-extra;libc' -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja -C /tmp/build libc-benchmark-main
```

> Note: The machine should run in `performance` mode. This is achieved by running:
```shell
cpupower frequency-set --governor performance
```

### Usage

`libc-benchmark-main` can run in two modes:
 - **stochastic mode** returns the average time per call for a particular size distribution,
 - **sweep mode** returns the average time per size over a range of sizes.

The tool requires the following flags to be set:
 - `--study-name`: a name to identify a run and provide label during analysis,
 - `--function`: the name of the function under test.

It also provides optional flags:
 - `--num-trials`: repeats the benchmark more times, the analysis tool can take this into account and give confidence intervals.
 - `--output`: specifies a file to write the report - or standard output if not set.
 - `--aligned-access`: The alignment to use when accessing the buffers, default is unaligned, 0 disables address randomization.

> Note: `--function` takes a generic function name like `memcpy` or `memset` but the actual function being tested is the llvm-libc implementation (e.g. `__llvm_libc::memcpy`).

### Stochastic mode

This is the preferred mode to use. The function parameters are randomized and the branch predictor is less likely to kick in.

```shell
/tmp/build/bin/libc-benchmark-main \
    --study-name="new memcpy" \
    --function=memcpy \
    --size-distribution-name="memcpy Google A" \
    --num-trials=30 \
    --output=/tmp/benchmark_result.json
```

The `--size-distribution-name` flag is mandatory and points to one of the [predefined distribution](MemorySizeDistributions.h).

> Note: These distributions are gathered from several important binaries at Google (servers, databases, realtime and batch jobs) and reflect the importance of focusing on small sizes.

Using a profiler to observe size distributions for calls into libc functions, it
was found most operations act on a small number of bytes.

Function           | % of calls with size ≤ 128 | % of calls with size ≤ 1024
------------------ | --------------------------: | ---------------------------:
memcpy             | 96%                         | 99%
memset             | 91%                         | 99.9%
memcmp<sup>1</sup> | 99.5%                       | ~100%

_<sup>1</sup> - The size refers to the size of the buffers to compare and not
the number of bytes until the first difference._

### Sweep mode

This mode is used to measure call latency per size for a certain range of sizes. Because it exercises the same size over and over again the branch predictor can kick in. It can still be useful to compare strength and weaknesses of particular implementations.

```shell
/tmp/build/bin/libc-benchmark-main \
    --study-name="new memcpy" \
    --function=memcpy \
    --sweep-mode \
    --sweep-max-size=128 \
    --output=/tmp/benchmark_result.json
```

## Analysis tool

### Setup

Make sure to have `matplotlib`, `pandas` and `seaborn` setup correctly:

```shell
apt-get install python3-pip
pip3 install matplotlib pandas seaborn
```
You may need `python3-gtk` or similar package to display the graphs.

### Usage

```shell
python3 libc/benchmarks/libc-benchmark-analysis.py3 /tmp/benchmark_result.json ...
```

When used with __multiple trials Sweep Mode data__ the tool displays the 95% confidence interval.

When providing with multiple reports at the same time, all the graphs from the same machine are displayed side by side to allow for comparison.

The Y-axis unit can be changed via the `--mode` flag:
 - `time` displays the measured time (this is the default),
 - `cycles` displays the number of cycles computed from the cpu frequency,
 - `bytespercycle` displays the number of bytes per cycle (for `Sweep Mode` reports only).

## Under the hood

 To learn more about the design decisions behind the benchmarking framework,
 have a look at the [RATIONALE.md](RATIONALE.md) file.
