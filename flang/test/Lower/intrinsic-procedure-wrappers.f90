! RUN: bbc -outline-intrinsics %s -o - | tco --disable-llvm --mlir-print-ir-after=fir-to-llvm-ir 2>&1 | FileCheck %s

! Test properties of intrinsic function wrappers

! Test that intrinsic wrappers have internal linkage
function foo(x)
  foo = acos(x)
end function

! CHECK: llvm.func internal @fir.acos.contract.f32.f32
