# Hotswap Transpiler

The hotswap transpiler raises AMDGPU code objects into LLVM IR, re-lowers
them through the stock AMDGPU backend for a different target ISA, and
relinks the result into a single merged HSACO. It is a sibling to the
byte-level `amd_comgr_hotswap_rewrite` API: where rewrite applies a small
set of stepping-specific patches in place, transpilation hands the entire
code object to the IR pipeline.

## Build

The library can be configured standalone for development:

```
cmake -S amd/comgr/hotswap -B build-hotswap \
  -DLLVM_DIR=$PWD/build/lib/cmake/llvm
ninja -C build-hotswap
ctest --test-dir build-hotswap -L transpiler
```
