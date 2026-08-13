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
The default runtime device substring is `B60`, paired with IGC's `bmg-g21`
target. Override both `--device` and `--igc-device` together when targeting a
different GPU architecture.

The OpenCL reference mirrors the Lighthouse operations and cache policy. Its
2D prefetch builtins compile to cached `load_block2d.ugm.d16.a64.ca.ca`
messages, matching the cached prefetch contract in the Inter input.
