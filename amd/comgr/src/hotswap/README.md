# HotSwap

HotSwap is COMGR's AMDGPU code-object rewriting support. The public
`amd_comgr_hotswap_rewrite` API takes an executable code object plus source and
target ISA names, then returns a new executable code object with the applicable
rewrite applied. The input code object is not modified.

This directory contains COMGR's hotswap transpiler scaffolding, the raiser-based
path for heavier cross-ISA transformations. The same-family stepping patches are
implemented in the surrounding COMGR source files and are exposed through
`amd_comgr_hotswap_rewrite`.

## Transpiler (cross-gen)

The transpiler is the heavier sibling to the byte-level rewrite. It raises
AMDGPU code objects into LLVM IR, re-lowers them through the stock AMDGPU backend
for a different target ISA, and relinks the result into a single merged HSACO.
The rewrite path applies in-place stepping patches; the transpiler instead hands
the whole code object to the IR pipeline. It can be built standalone for
development:

```bash
cmake -S amd/comgr/hotswap -B build-hotswap \
  -DLLVM_DIR=$PWD/build/lib/cmake/llvm
ninja -C build-hotswap
ctest --test-dir build-hotswap -L transpiler
```
