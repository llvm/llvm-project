// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm \
// RUN:   -debug-info-kind=constructor -I %S/Inputs -isystem %S/SysInputs \
// RUN:   -o - %s | FileCheck %s

#include "a.hlsl"
#include "a.hlsl"
#include <c.hlsl>

// The main file appears first in dx.source.contents; included files are
// appended in sorted order afterwards. Duplicate and system includes are ignored.

// CHECK: !dx.source.contents = !{![[MAIN:[0-9]+]], ![[H1:[0-9]+]], ![[H2:[0-9]+]]}
// CHECK: !dx.source.mainFileName = !{![[MAIN_FILE_NAME:[0-9]+]]}
// CHECK: ![[MAIN]] = !{!"{{.*[\\/]dx-source-metadata-includes.hlsl}}", !"{{.*}}"}
// CHECK: ![[H1]] = !{!"{{.*[\\/]a.hlsl}}", !"{{.*}}"}
// CHECK: ![[H2]] = !{!"{{.*[\\/]b.hlsl}}", !"{{.*}}"}
// CHECK: ![[MAIN_FILE_NAME]] = !{!"{{.*[\\/]dx-source-metadata-includes.hlsl}}"}

float foo(float a, float b) {
  return helper1_add(a, b) + helper2_mul(a, b) + sys_helper(a, b);
}
