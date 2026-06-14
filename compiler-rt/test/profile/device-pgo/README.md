# HIP device PGO / code coverage: standalone build & test recipe

This directory provides a CMake-based recipe to build and exercise HIP device
profile-guided optimization (PGO) and source-based code coverage **outside
TheRock**, using only an `llvm-project` checkout plus a ROCm runtime.

It builds, in one configure:

- the host toolchain (`clang`, `clang++`, `lld`, `llvm-profdata`, `llvm-cov`)
  and the lit-lite test utilities (`FileCheck`, `not`);
- the host ROCm drain runtime `clang_rt.profile_rocm` (opt-in,
  `COMPILER_RT_BUILD_PROFILE_ROCM=ON`). It runs the upstream host-shadow drain on
  all platforms; on **Linux** `InstrProfilingPlatformROCm.cpp` additionally runs a
  supplemental HSA-introspection pass (with content-dedup) to collect device code
  objects that have no host shadow (e.g. device-linked/RCCL kernels);
- the **amdgcn device** profile runtime `libclang_rt.profile.a` (the baremetal
  profile subset that provides `__llvm_profile_instrument_gpu` and the
  `__llvm_profile_sections` bounds table), built for the `amdgcn-amd-amdhsa`
  runtime target with LLVM libc for amdgcn.

## Why a separate library

Upstream relands HIP offload PGO runtime support as the **opt-in**
`clang_rt.profile_rocm` (llvm#201606), a `/MD` superset of `clang_rt.profile`;
the base library stays unchanged. The driver links `clang_rt.profile_rocm`
ahead of `clang_rt.profile` for HIP host links when profiling is requested
(see `clang/lib/Driver/ToolChains/{Linux,MSVC}.cpp`). This recipe just turns
the option on and builds the matching amdgcn device runtime.

## Prerequisites

- A ROCm installation (for `libamdhip64` and, on Linux, `libhsa-runtime64`),
  e.g. `/opt/rocm`. Export `ROCM_PATH`.
- An AMD GPU visible to the runtime for the *run* step (the build step does
  not need a GPU). `amdgpu-arch` should list your device(s).
- Ninja, a host C/C++ compiler, and Python 3.

## Build

```bash
export ROCM_PATH=/opt/rocm
./build.sh                 # builds into <repo>/build/device-pgo
# or: ./build.sh /path/to/builddir
```

Key outputs under the build dir:

```
bin/{clang,clang++,lld,llvm-profdata,llvm-cov,FileCheck,not}
lib/clang/<ver>/lib/<host-triple>/libclang_rt.profile_rocm.a
lib/clang/<ver>/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a
```

See `toolchain-cache.cmake` for the exact CMake variables, including the
`LLVM_RUNTIME_TARGETS="default;amdgcn-amd-amdhsa"` split.

## Run the tests

The lit-lite runner (`../run_gpu_tests.py`) compiles each `.hip` test with the
just-built toolchain, runs it on the GPU, and pipes output through `FileCheck`.
It auto-detects features (`multi-device` via `amdgpu-arch`) so tests that need
two visible GPUs are skipped on single-GPU hosts.

```bash
python3 ../run_gpu_tests.py \
    --toolchain-bin "$PWD/<builddir>/bin" \
    --hip-lib-path "$ROCM_PATH/lib" \
    ../GPU ../AMDGPU
```

`--toolchain-bin` must be an **absolute** path (the runner executes each RUN
line from a temp directory). With the toolchain's `amdgpu-arch`/`offload-arch`
on hand, `--offload-arch=native` resolves automatically and the `multi-device`
feature is enabled when 2+ GPUs are visible (so multi-GPU tests run on a
multi-GPU host and are skipped otherwise). On a multi-gfx90a host this suite is
15 passed, 0 failed.

### Coverage notes / known gaps

- Quantitative device-counter correctness (`instrprof-hip-counter-correctness`),
  multi-process offline accumulation (`instrprof-hip-multi-process-merge`) and
  explicit-collect idempotency (`instrprof-hip-collect-after`) pin exact device
  counter values, so a dedup or drain regression that drops/doubles a section is
  caught.
- `LLVM_PROFILE_FILE=...%m` on-the-fly merge-pooling does **not** accumulate
  *device* counters today (the device profraw is rewritten per process rather
  than merged in place); multi-process accumulation must go through
  `llvm-profdata merge` of distinct per-process files.
- There is no in-tree test that drains a code object with **no** host shadow in
  isolation (the pure device-linked/RCCL case the HSA pass uniquely handles): it
  requires a real device-side library build (the profile runtime linked into the
  device image), which is not expressible in the lit-lite harness via the clang
  driver. The dedup tests do prove the HSA pass finds and dedups the same code
  objects the host-shadow pass drains; validating the no-host-shadow drain needs
  an actual RCCL-style binary in downstream CI.

## Manual workflow (for reference)

```bash
CLANG=<builddir>/bin/clang++
# 1. Instrumented build (host + device).
$CLANG -O2 -fprofile-instr-generate -fcoverage-mapping \
    --offload-arch=gfx1100 -xhip app.hip -o app

# 2. Run. Produces a host .profraw and a device
#    <name>.amdgcn-amd-amdhsa.<arch>.profraw drained by clang_rt.profile_rocm.
LLVM_PROFILE_FILE='app-%p.profraw' ./app

# 3. Merge (device profiles are merged per GPU arch).
<builddir>/bin/llvm-profdata merge -o app.profdata app-*.profraw

# 4. Coverage report (device).
<builddir>/bin/llvm-cov show ./app -instr-profile=app.profdata
```

## Notes / environment-specific knobs

- `--offload-arch` must match your GPU; the amdgcn device runtime is target
  generic but the app's device code is per-arch. The build installs
  `offload-arch` (and the `amdgpu-arch` alias) into `<builddir>/bin`, so
  `--offload-arch=native` works without a system ROCm `amdgpu-arch`.
- The amdgcn runtime target requires LLVM libc for amdgcn; if your environment
  cannot build it, drop `libc` from
  `RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES` only if your headers are
  otherwise provided.
