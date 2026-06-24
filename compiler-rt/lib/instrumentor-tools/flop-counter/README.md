# FLOP Counter

A runtime library for counting floating-point operations in programs using the LLVM Instrumentor pass.

## Features

- **Precision Tracking**: Separates counts for single (float), double, and extended precision operations
- **Operation Categorization**: Tracks adds, multiplications, divisions, FMA operations (TODO), and others (sqrt, sin, cos, etc.) (TODO)
- **Vector Support**: Counts FLOPs in vector operations
- **Thread-Safe**: Uses atomic operations for counter updates
- **Low Overhead**: Minimal runtime overhead for counting
- **Automatic Reporting**: Prints statistics at program exit

## Usage

### Basic Example

```c
#include <stdio.h>
#include <math.h>

double compute(double a, double b) {
  return sqrt(a * a + b * b);
}

int main() {
  double result = compute(3.0, 4.0);
  printf("Result: %f\n", result);
  return 0;
}
```

Compile with:
```bash
clangxx -O2 -finstrumentor=flop_counter_config.json example.cpp \
        -lclang_rt.flop_counter -o example
```

Run:
```bash
./example
```

Output:
```
Result: 5.000000

=================================================
           FLOP Counter Statistics
=================================================
Total FLOPs:                             3
...
```

## Implementation Details

### Instrumentation Points

The FLOP counter instruments:

1. **Binary FP Operations**: `fadd`, `fsub`, `fmul`, `fdiv`, `frem`
2. **Unary FP Operations**: `fneg`
3. TODO: **FP Intrinsics**: `llvm.fma`, `llvm.sqrt`, `llvm.sin`, `llvm.cos`, etc.

### FLOP Counting Rules

- **Regular operations**: 1 FLOP per operation
- **FMA (Fused Multiply-Add)**: 2 FLOPs (multiply + add)
- **Vector operations**: Counted per element
- **Intrinsics**: TODO

### Configuration

The `flop_counter_config.json` file configures the instrumentor to:
- Insert callbacks after floating-point binary/unary operations
- Pass value size, type IDs, and opcodes to the runtime
- Filter to only instrument FP math operations
