# Inter Development Rules

## Dialect Metadata

- Define operation metadata structurally in ODS. Do not attach declared
  operation fields with raw `setAttr` calls.
- Access ODS fields through generated operation accessors. Do not look up
  operation fields by string name in handwritten C++.
- Do not introduce constants for internal ODS field names. A named string
  constant does not make a stringly-typed operation contract structural.
- When multiple operation kinds expose the same semantic property, define a
  dialect operation interface and consume that interface outside the dialect.
- Interfaces must expose both reads and mutations needed by consumers. Do not
  bypass an interface with raw attribute access for updates.
- Keep serialized metadata names centralized in the owning dialect when no
  generated structural accessor can exist.

## Hardware Integration

- Configure hardware tests with `-DINTER_INCLUDE_INTEGRATION_TESTS=ON` and
  select the device with `-DINTER_INTEGRATION_DEVICE_NAME=<substring>`.
- Run hardware tests serially. Use `ninja -C <build> check-inter-integration`
  for the full suite or `llvm-lit -sv -j 1 <test>` for one test. Concurrent GPU
  stress can produce `ZE_RESULT_ERROR_DEVICE_LOST` and invalidates the result.
- A single successful launch is not sufficient for kernels with persistent
  physical-register or synchronization hazards. Stress repeated launches when
  changing predication, region transport, loop-carried values, SWSB, or EOT.
- Keep host-only tests available through `ninja -C <build> check-inter`; do not
  make them depend on a GPU, Level Zero, or IGC.

## Benchmarking

- Use `benchmarks/lighthouse.py` for the Inter-versus-Lighthouse comparison.
  Build `inter-opt`, `inter-translate`, and `inter-lighthouse-benchmark`, then
  run `python3 inter/benchmarks/lighthouse.py --build-dir <build>`.
- Benchmark runs must be serialized and correctness-checked. Do not report
  timing from a kernel that fails any warmup or final output validation.
- Compare identical problem sizes, workgroup geometry, warmup count, batch
  count, iteration count, and Level Zero timestamp source. Alternate compiler
  order across repeated runs to reduce temperature and clock-order bias.
- Generate the reference binary through the pinned Lighthouse MLIR pipeline.
  Do not substitute an OpenCL reconstruction of the kernel.
- Report the distribution and method, not one favorable sample. At minimum,
  include median latency, observed range, run count, batches, iterations, and
  relative Inter/Lighthouse performance.
- Regenerate both compiler outputs from source for a comparison. Do not use
  stale binaries from `/tmp`, previous commits, or a different driver/compiler
  installation without explicitly identifying that mismatch.
