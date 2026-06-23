# Pointer Tracking Example

This example demonstrates using the LLVM Instrumentor pass to track memory
allocations and their usage patterns. It replaces real pointers with "fake
pointers" that encode allocation IDs and maintains a global table of metadata
for each allocation.

## Features

- **Alloca Instrumentation**: Tracks stack allocations after they are created
- **Global Instrumentation**: Tracks global variables after initialization
- **Load/Store Instrumentation**: Instruments loads and stores before execution
  to repair pointers and track read/write patterns
- **Function Exit Instrumentation**: Clears stack allocations and computes timing
- **Fake Pointer Scheme**: Replaces pointers with fake pointers encoding allocation IDs
- **Metadata Tracking**: For each allocation, tracks:
  - Whether it was read
  - Whether it was written
  - Timestamps for allocation, first use, and last use
- **Timing Analysis**:
  - Time from allocation to first use
  - Time from last use to deallocation (unused timer)
- **Statistics Reporting**: Reports usage patterns and top 5 longest timing records

## Configuration

The instrumentation is configured via `pointer_tracking_config.json`:

- `alloca_post`: Instruments allocas after creation, replaces the pointer
- `global_pre`: Instruments globals after creation, replaces the pointer
- `load_pre`: Instruments loads before execution, replaces the pointer operand
- `store_pre`: Instruments stores before execution, replaces the pointer operand
- `function_post`: Instruments function exits to clear stack allocations

## Building

The runtime library is built as part of the compiler-rt build:

```bash
ninja -C build/runtimes/runtimes-bins pointer-tracking
```

## Usage

Compile your program with the Instrumentor pass enabled and link with the runtime:

```bash
clang++ -O0 -g \
  -mllvm -enable-instrumentor \
  -mllvm -instrumentor-read-config-files=pointer_tracking_config.json \
  your_program.c \
  -lclang_rt.pointer_tracking \
  -o your_program

./your_program
```

## Example Output

```
=================================================
        Pointer Tracking Statistics
=================================================
Total allocations tracked: 15

Usage Patterns:
  Read-only:                        3
  Write-only:                       2
  Read and Write:                   8
  Unused:                           2

Top 5 Longest Unused Times (last use to deallocation):
  Allocation ID   1234:       123456 ns
  Allocation ID   5678:        98765 ns
  Allocation ID   9012:        87654 ns
  Allocation ID   3456:        76543 ns
  Allocation ID   7890:        65432 ns

Top 5 Longest First Use Times (allocation to first use):
  Allocation ID   2345:       234567 ns
  Allocation ID   6789:       198765 ns
  Allocation ID   0123:       187654 ns
  Allocation ID   4567:       176543 ns
  Allocation ID   8901:       165432 ns
=================================================
```
