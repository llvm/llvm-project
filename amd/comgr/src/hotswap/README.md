# HotSwap

HotSwap is COMGR's AMDGPU code-object rewriting support. The public
`amd_comgr_hotswap_rewrite` API takes an executable code object plus source and
target ISA names, then returns a new executable code object with the applicable
rewrite applied. The input code object is not modified.

This directory contains COMGR's hotswap transpiler scaffolding, the raiser-based
path for heavier cross-ISA transformations. The same-family stepping patches and
entry trampolines are implemented in the surrounding COMGR source files and are
exposed through `amd_comgr_hotswap_rewrite_with_options`.

## Supported transformations

| Transformation | Status |
| -------------- | ------ |
| gfx1250 B0 to A0 | Supported |
| gfx125x entry trampolines | Supported, opt-in |
| gfx950 | Coming soon |
| gfx942 | Coming soon |

## Rewrite options

Callers request optional gfx125x kernel descriptor entry redirection through
`amd_comgr_hotswap_rewrite_with_options` with
`AMD_COMGR_HOTSWAP_REWRITE_FLAG_ENTRY_TRAMPOLINES`.

Callers request opt-in B0 strict-mode mask workarounds through
`AMD_COMGR_HOTSWAP_REWRITE_FLAG_STRICT_MODE`. If a required selected mask
workaround is detected but cannot be emitted safely, the rewrite fails instead
of returning the original unpatched code object.

`AMD_COMGR_STATUS_SUCCESS` means COMGR produced a valid output code object, not
necessarily that the output bytes changed. If the source/target ISA pair and
rewrite options select no enabled transformation, the output is a copy of the
input.

## Register accounting

A patch that allocates VGPRs above the kernel descriptor's existing allocation
is checked before any replacement bytes are emitted. The proposed allocation,
rounded to the target and kernel wavefront mode's VGPR granule, must retain
enough waves per execution unit to admit one workgroup at the kernel metadata's
`.max_flat_workgroup_size`. Optional patches are declined when that invariant
would be violated; required patches fail the rewrite. This also protects
cluster dispatches, whose constituent workgroups must each remain schedulable.

When a VGPR bump is committed, hotswap updates both the kernel descriptor and
the runtime-visible `.vgpr_count` metadata. Missing or malformed workgroup
metadata is not treated as permission to grow the allocation. SGPR and LDS
usage are outside this differential check: SGPRs are not occupancy-limiting on
the supported gfx125x path, and current hotswap patches do not increase LDS.

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
