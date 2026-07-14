// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan1.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,SPIRV
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,DX
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan1.3-library %s \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s --check-prefix=SPIRV-OPT
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s --check-prefix=DX-OPT

// When Targeting SPIR-V Non-entry, non-exported HLSL functions are always 
// inlined making unused definitions dead and removable by DCE.
// On DX the function keeps external linkage with hidden visibility.

// Exported functions must remain externally visible on all targets.
// CHECK: define {{(spir_func )?}}noundef nofpclass(nan inf) float @_Z11exported_fnf(
export float exported_fn(float x) { return x * 2.0; }

// SPIRV: define internal spir_func noundef nofpclass(nan inf) float @_Z6helperf(
// DX: define hidden noundef nofpclass(nan inf) float @_Z6helperf(
float helper(float x) { return x * 3.0; }

// The mangled entry implementation always has internal linkage.
// CHECK: define internal {{(spir_func )?}}void @_Z4mainv()

// The unmangled entry point wrapper is always externally visible.
// CHECK: define void @main()
[shader("compute")][numthreads(1,1,1)]
void main() {}

// The dead, internalized helper is removed on SPIR-V once passes run, while the
// externally-visible exported function and entry point are retained.
// SPIRV-OPT-NOT: @_Z6helperf
// SPIRV-OPT: define spir_func noundef nofpclass(nan inf) float @_Z11exported_fnf(
// SPIRV-OPT: define void @main()

// For DirectX the helper is removed in the backend and so should persist here.
// DX-OPT: define hidden noundef nofpclass(nan inf) float @_Z6helperf(
