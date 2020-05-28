# Libc mem* benchmarks

This framework has been designed to evaluate and compare relative performance of
memory function implementations on a particular host.

It will also be use to track implementations performances over time.

## Quick start

### Setup

**Python 2** [being deprecated](https://www.python.org/doc/sunset-python-2/) it is
advised to used **Python 3**.

Then make sure to have `matplotlib`, `scipy` and `numpy` setup correctly:

```shell
apt-get install python3-pip
pip3 install matplotlib scipy numpy
```
You may need `python3-gtk` or similar package for displaying benchmark results.

To get good reproducibility it is important to make sure that the system runs in
`performance` mode. This is achieved by running:

```shell
cpupower frequency-set --governor performance
```

### Run and display `memcpy` benchmark

The following commands will run the benchmark and display a 95 percentile
confidence interval curve of **time per copied bytes**. It also features **host
informations** and **benchmarking configuration**.

```shell
cd llvm-project
cmake -B/tmp/build -Sllvm -DLLVM_ENABLE_PROJECTS='clang;clang-tools-extra;libc' -DCMAKE_BUILD_TYPE=Release
make -C /tmp/build -j display-libc-memcpy-benchmark-small
```

The display target will attempt to open a window on the machine where you're
running the benchmark. If this may not work for you then you may want `render`
or `run` instead as detailed below.

## Benchmarking targets

The benchmarking process occurs in two steps:

1. Benchmark the functions and produce a `json` file
2. Display (or renders) the `json` file

Targets are of the form `<action>-libc-<function>-benchmark-<configuration>`

 - `action` is one of :
    - `run`, runs the benchmark and writes the `json` file
    - `display`, displays the graph on screen
    - `render`, renders the graph on disk as a `png` file
 - `function` is one of : `memcpy`, `memcmp`, `memset`
 - `configuration` is one of : `small`, `big`

## Benchmarking regimes

Using a profiler to observe size distributions for calls into libc functions, it
was found most operations act on a small number of bytes.

Function           | % of calls with size ≤ 128 | % of calls with size ≤ 1024
------------------ | --------------------------: | ---------------------------:
memcpy             | 96%                         | 99%
memset             | 91%                         | 99.9%
memcmp<sup>1</sup> | 99.5%                       | ~100%

Benchmarking configurations come in two flavors:

 - [small](libc/utils/benchmarks/configuration_small.json)
    - Exercises sizes up to `1KiB`, representative of normal usage
    - The data is kept in the `L1` cache to prevent measuring the memory
      subsystem
 - [big](libc/utils/benchmarks/configuration_big.json)
    - Exercises sizes up to `32MiB` to test large operations
    - Caching effects can show up here which prevents comparing different hosts

_<sup>1</sup> - The size refers to the size of the buffers to compare and not
the number of bytes until the first difference._

## Superposing curves

It is possible to **merge** several `json` files into a single graph. This is
useful to **compare** implementations.

In the following example we superpose the curves for `memcpy`, `memset` and
`memcmp`:

```shell
> make -C /tmp/build run-libc-memcpy-benchmark-small run-libc-memcmp-benchmark-small run-libc-memset-benchmark-small
> python libc/utils/benchmarks/render.py3 /tmp/last-libc-memcpy-benchmark-small.json /tmp/last-libc-memcmp-benchmark-small.json /tmp/last-libc-memset-benchmark-small.json
```

## Useful `render.py3` flags

 - To save the produced graph `--output=/tmp/benchmark_curve.png`.
 - To prevent the graph from appearing on the screen `--headless`.


## Under the hood

 To learn more about the design decisions behind the benchmarking framework,
 have a look at the [RATIONALE.md](RATIONALE.md) file.
