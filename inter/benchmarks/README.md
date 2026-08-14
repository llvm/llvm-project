# Lighthouse performance comparison

This benchmark compiles the same `128x128x64` f16-to-f32 Lighthouse matmul
through Inter and IGC, validates repeated launches, and compares Level Zero
kernel timestamps using identical `2x2` workgroups of 256 threads.

Configure Inter with hardware integration enabled, build the benchmark, and run
the orchestration script:

```sh
cmake -S inter -B inter/build -G Ninja \
  -DMLIR_DIR=$PWD/build-m0/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/build-m0/lib/cmake/llvm \
  -DINTER_INCLUDE_INTEGRATION_TESTS=ON
ninja -C inter/build inter-opt inter-translate inter-lighthouse-benchmark
python3 inter/benchmarks/lighthouse.py --build-dir inter/build
```

Defaults are five alternating Inter/IGC runs. Each run performs 200 validated
warmup launches followed by 15 batches of 1,000 timestamped launches. Use
`--runs`, `--warmups`, `--batches`, and `--iterations` for shorter experiments.
Each compiler run has a 120-second timeout; override it with `--timeout`.
Inputs are deterministic dense `{-1, 0, +1}` matrices. Validation combines
1,024 exact CPU dot products, complete finite/integer/output-bound scans, two
independent Freivalds projections, and canaries around C. Between timed batches,
an untimed launch writes a freshly poisoned C allocation; another untimed launch
restores GPU residency after host validation.
The default runtime device substring is `B60`, paired with IGC's `bmg-g21`
target. Override both `--device` and `--igc-device` together when targeting a
different GPU architecture.

Use `--size 256` (or another multiple of 64) to scale the square M/N dimensions
while retaining `K=64` and the same `64x64` workgroup tile. This isolates grid
scaling from reduction-loop scaling.

Use `--reduction-size 256` (or another multiple of 32) to scale K independently.
The script updates A/B surfaces and the Inter/IGC loop bounds consistently.

The OpenCL reference mirrors the Lighthouse operations and cache policy. Its
2D prefetch builtins compile to cached `load_block2d.ugm.d16.a64.ca.ca`
messages, matching the cached prefetch contract in the Inter input.

Use `--drop-loop-prefetch` to remove the two loop-ahead prefetches from both
compiler inputs and measure their performance impact. The final one-past
prefetch pair is ignored by the 2D block I/O contract.
Use `--padding-k-tiles 1` to extend the physical A/B allocations without
changing their logical block2D descriptor dimensions.
