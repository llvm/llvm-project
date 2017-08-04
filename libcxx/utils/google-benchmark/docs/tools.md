# Benchmark Tools

## compare_bench.py

The `compare_bench.py` utility which can be used to compare the result of benchmarks.
The program is invoked like:

``` bash
$ compare_bench.py <old-benchmark> <new-benchmark> [benchmark options]...
```

Where `<old-benchmark>` and `<new-benchmark>` either specify a benchmark executable file, or a JSON output file. The type of the input file is automatically detected. If a benchmark executable is specified then the benchmark is run to obtain the results. Otherwise the results are simply loaded from the output file.

The sample output using the JSON test files under `Inputs/` gives:

``` bash
$ ./compare_bench.py ./gbench/Inputs/test1_run1.json ./gbench/Inputs/test1_run2.json
Comparing ./gbench/Inputs/test1_run1.json to ./gbench/Inputs/test1_run2.json
Benchmark                   Time           CPU
----------------------------------------------
BM_SameTimes               +0.00         +0.00
BM_2xFaster                -0.50         -0.50
BM_2xSlower                +1.00         +1.00
BM_10PercentFaster         -0.10         -0.10
BM_10PercentSlower         +0.10         +0.10
```

When a benchmark executable is run, the raw output from the benchmark is printed in real time to stdout. The sample output using `benchmark/basic_test` for both arguments looks like:

```
./compare_bench.py  test/basic_test test/basic_test  --benchmark_filter=BM_empty.*
RUNNING: test/basic_test --benchmark_filter=BM_empty.*
Run on (4 X 4228.32 MHz CPU s)
2016-08-02 19:21:33
Benchmark                              Time           CPU Iterations
--------------------------------------------------------------------
BM_empty                               9 ns          9 ns   79545455
BM_empty/threads:4                     4 ns          9 ns   75268816
BM_empty_stop_start                    8 ns          8 ns   83333333
BM_empty_stop_start/threads:4          3 ns          8 ns   83333332
RUNNING: test/basic_test --benchmark_filter=BM_empty.*
Run on (4 X 4228.32 MHz CPU s)
2016-08-02 19:21:35
Benchmark                              Time           CPU Iterations
--------------------------------------------------------------------
BM_empty                               9 ns          9 ns   76086957
BM_empty/threads:4                     4 ns          9 ns   76086956
BM_empty_stop_start                    8 ns          8 ns   87500000
BM_empty_stop_start/threads:4          3 ns          8 ns   88607596
Comparing test/basic_test to test/basic_test
Benchmark                              Time           CPU
---------------------------------------------------------
BM_empty                              +0.00         +0.00
BM_empty/threads:4                    +0.00         +0.00
BM_empty_stop_start                   +0.00         +0.00
BM_empty_stop_start/threads:4         +0.00         +0.00
```

Obviously this example doesn't give any useful output, but it's intended to show the output format when 'compare_bench.py' needs to run benchmarks.
