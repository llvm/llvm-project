# gen_icache_stress.py

A Python script that generates a large C program designed to stress the CPU's instruction cache (I-cache) and Branch Target Buffer (BTB), with realistic three-tier hot/warm/cold code distribution.

## Usage

```bash
python gen_icache_stress.py [options]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-functions N` | 5000 | Total number of compute functions |
| `--hot-ratio R` | 0.2 | Ratio of hot functions (default: 20%) |
| `--warm-ratio R` | 0.4 | Ratio of warm functions (default: 40%) |
| `--num-switch N` | 50 | Number of switch-based functions |
| `--num-goto N` | 50 | Number of computed goto functions |
| `--branches-per-func N` | 15 | Branches per compute function |
| `--iterations N` | 150000000 | Number of loop iterations |
| `--seed N` | 42 | Random seed for code generation reproducibility |
| `--use-func-ptrs` | off | Enable function pointer indirect calls |
| `--no-switch` | | Disable switch functions |
| `--no-goto` | | Disable computed goto functions |
| `-o, --output FILE` | icache_stress.c | Output filename |

## Examples

```bash
# Default configuration (20/40/40 hot/warm/cold)
python gen_icache_stress.py

# More functions
python gen_icache_stress.py --num-functions 1000

# Minimal (only compute functions)
python gen_icache_stress.py --no-switch --no-goto

# Maximum BTB stress with indirect calls
python gen_icache_stress.py --use-func-ptrs

# Custom output file
python gen_icache_stress.py -o my_stress_test.c
```

## Compiling and Running

```bash
# Compile
gcc -O2 -o icache_stress icache_stress.c

# Run with default seed (42)
./icache_stress

# Run with custom seed
./icache_stress 12345
```

## Hot/Warm/Cold Code Distribution

The generated code models realistic application behavior with three tiers:

### Function level (20/40/40)
- **Hot functions** (20%): Called every iteration
- **Warm functions** (40%): Called rarely (~0.01% of iterations)
- **Cold functions** (40%): Never called at runtime (exist in binary but unreachable)

### Within hot functions (40/25/35)
Each hot function contains branches distributed as:
- **Hot code** (40%): Both branch paths are executed (based on data)
- **Warm code** (25%): Rarely executed (~0.1% probability)
- **Cold code** (35%): Never executed (condition is always false)

### Verification
The program prints checksums at exit:
- `Result checksum`: Main computation result
- `Warm path checksum`: Sum from warm code paths (should be non-zero but small)
- `Cold path checksum`: Sum from cold code paths (should always be 0)

## Determinism

The generated program is fully deterministic:
- The Python script uses `--seed` to reproducibly generate the C code
- The C program accepts an optional seed argument for reproducible runtime behavior
- A fixed iteration count ensures consistent execution length

## How It Works

The generated program contains several types of functions:

1. **Hot compute functions**: Called every iteration. Each contains branches with 40/25/35 hot/warm/cold code distribution. Cold paths use conditions that are always false at runtime.

2. **Warm functions**: Called very rarely (~0.01% of iterations). Simulate occasional error handling or edge case processing.

3. **Cold functions**: Never called at runtime. Exist in the binary to simulate dead code, unused error handlers, and rarely-needed functionality.

4. **Switch functions**: Large switch statements with many cases, causing indirect jumps through jump tables.

5. **Computed goto functions**: Use GCC's computed goto extension (`goto *label_ptr`) for unpredictable indirect branches.

## Use Cases

- Benchmarking BOLT optimizations on I-cache-bound workloads
- Testing function reordering and code layout optimizations
- Measuring the impact of hot/cold code separation
- Testing dead code elimination and function splitting
- Generating realistic workloads for performance analysis
