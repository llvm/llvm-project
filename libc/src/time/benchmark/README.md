# Time Conversion Benchmark

This benchmark compares the performance of two time conversion implementations:

1. **`update_from_seconds_fast`** - Howard Hinnant's civil_from_days algorithm with March-based year optimization (in `time_utils.cpp`)
2. **`unix_to_date_fast`** - Ben Joffe's Century-February-Padding algorithm (in `fast_date.cpp`)

## Quick Start

```bash
cd /workspaces/cpp-experiments/libc/src/time/benchmark
make -f Makefile.standalone run
```

## Building Options

### Option 1: Standalone Build (Recommended - No Dependencies)
```bash
make -f Makefile.standalone
./benchmark_time_conversion
```

This uses the standalone version with all code inline.

### Option 2: Using CMake
```bash
cd /workspaces/cpp-experiments/libc/src/time/benchmark
./build.sh
```

### Option 3: Manual Compilation
```bash
cd /workspaces/cpp-experiments/libc/src/time/benchmark
g++ -std=c++17 -O3 -march=native \
    -I.. -I../../.. -I../../../include -I../../../hdr \
    -DLIBC_NAMESPACE=__llvm_libc -DLIBC_NAMESPACE_DECL=__llvm_libc \
    benchmark_time_conversion.cpp ../time_utils.cpp ../fast_date.cpp \
    -o benchmark_time_conversion
./benchmark_time_conversion
```

## What It Tests

The benchmark:
- **Generates diverse test cases**: Unix epoch, leap years, century boundaries, negative timestamps, random dates
- **Verifies correctness**: Ensures both functions produce identical results
- **Measures performance**: Times both implementations over 1 million iterations
- **Reports speedup**: Shows which function is faster and by how much

## Expected Output

```
=== Time Conversion Benchmark ===

Generated 70 test timestamps

Verifying correctness...
✓ All results match!

Warming up (10000 iterations)...
Warmup complete

Running benchmarks (1000000 iterations)...

update_from_seconds_fast: 14.23 ns/conversion
unix_to_date_fast:        12.87 ns/conversion

=== Results ===
unix_to_date_fast is 1.11x FASTER (9.6% improvement)
```

## Implementation Details

Both functions implement similar algorithms but with different approaches:

- **`update_from_seconds_fast`**: Uses Howard Hinnant's algorithm with a March-based year (February as last month)
- **`unix_to_date_fast`**: Uses Ben Joffe's algorithm with proleptic Gregorian calendar mapping

The benchmark helps determine which implementation is more efficient for the LLVM libc project.
