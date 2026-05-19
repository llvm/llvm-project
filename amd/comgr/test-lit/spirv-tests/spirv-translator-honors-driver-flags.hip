// REQUIRES: comgr-has-spirv-translator
// REQUIRES: stability
// COM: Verify COMGR honors the --spirv-* flags HIPAMD emits today (see
// COM: clang/lib/Driver/ToolChains/HIPAMD.cpp). Each RUN pins one observable
// COM: side-effect of a specific driver flag. The kernel calls an amdgcn
// COM: intrinsic to exercise --spirv-allow-unknown-intrinsics, and -g is
// COM: passed to exercise --spirv-debug-info-version.

// RUN: source-to-spirv -nogpuinc -g %s %t.spv
// RUN: test -s %t.spv

// COM: --spirv-ext=+all,-SPV_KHR_untyped_pointers (disabled).
// RUN: not grep -a "SPV_KHR_untyped_pointers" %t.spv

// COM: --spirv-ext=+all (must allow extensions outside the default set).
// RUN: grep -a "SPV_INTEL_kernel_attributes" %t.spv

// COM: --spirv-allow-unknown-intrinsics (bare flag → allow all).
// RUN: grep -a "llvm.amdgcn.workitem.id.x" %t.spv

// COM: --spirv-preserve-auxdata (always-on extended instruction set import).
// RUN: grep -a "NonSemantic.AuxData" %t.spv

// COM: --spirv-debug-info-version=nonsemantic-shader-200.
// RUN: grep -a "NonSemantic.Shader.DebugInfo.200" %t.spv

// COM: Two of HIPAMD's driver flags have no test here:
// COM:  --spirv-max-version=1.6: TranslatorOpts::MaxVersion already defaults
// COM:    to MaximumVersion (1.6), so dropping the flag does not change the
// COM:    emitted version for any kernel that fits in <=1.6.
// COM:  --spirv-lower-const-expr: the global cl::opt in SPIRVLowerConstExpr.cpp
// COM:    is cl::init(true), so the flag matches the default and has no
// COM:    observable side-effect to assert on.

__attribute__((device))
unsigned get_id() { return __builtin_amdgcn_workitem_id_x(); }

__attribute__((global))
void k(unsigned* p) { *p = get_id(); }
