// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s  \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// FIXME: add semantic to a.
// See https://github.com/llvm/llvm-project/issues/57874
struct S {
  float a;
};


// Make sure sret parameter is generated.
// CHECK:define internal void @_Z7ps_mainv(ptr dead_on_unwind noalias writable sret(%struct.S) align 4 %agg.result)
// FIXME: change it to real value instead of poison value once semantic is add to a.
// Make sure the function with sret is called.
// CHECK:call void @_Z7ps_mainv(ptr poison)
[shader("pixel")]
S ps_main() {
  S s;
  s.a = 0;
  return s;
};
