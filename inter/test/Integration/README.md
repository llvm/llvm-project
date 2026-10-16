# Inter integration tests

Each `.ll` test contains its LLVM IR input, exposes the Inter compilation stages
in its `RUN` lines, emits a complete zebin, invokes `inter-runner`, and checks
the device result with `FileCheck` or the shared expression verifier. The M1-M3
kernels also run at 128 lanes to cover multiple hardware workgroups.
Allocator-generated scratch spill/fill code has a dedicated live-device test.
`opencl-smoke.cl` compiles a reference kernel with IGC and provides a quick
device-health check independent of Inter code generation.

Build the LLVM offload runtime first, then enable the opt-in suite when
configuring Inter:

```sh
ninja -C build-m0 offload
cmake -S inter -B inter/build -G Ninja \
  -DMLIR_DIR=$PWD/build-m0/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/build-m0/lib/cmake/llvm \
  -DINTER_INCLUDE_INTEGRATION_TESTS=ON
ninja -C inter/build check-inter-integration
```

`check-inter` remains host-only. The integration target builds
`inter-runner`, probes the selected Level Zero device, and serializes device
execution. Tests are reported as unsupported when the configured device is not
available.

Configuration variables:

- `INTER_OFFLOAD_INCLUDE_DIR`: directory containing `OffloadAPI.h`.
- `INTER_OFFLOAD_LIBRARY`: path to `libLLVMOffload`.
- `INTER_INTEGRATION_DEVICE_NAME`: required runtime device-name substring,
  default `B60`.

All generated files live below `inter/build/test/Integration/Output`; tests do
not share the source-tree `inter/out` directory.
