! RUN: bbc -emit-llvm -outline-intrinsics %s -o - | FileCheck %s

! Test properties of intrinsic function wrappers

! Test that intrinsic wrappers have internal linkage
function foo(x)
  foo = acos(x)
end function

! CHECK: llvm.func internal @fir.acos.f32.f32


! TODO: test wrapper mangling, attributes ...
