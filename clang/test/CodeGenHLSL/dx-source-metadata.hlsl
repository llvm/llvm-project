// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm \
// RUN:   -debug-info-kind=constructor -DUSER_DEF0=42 -DUSER_DEF1=43 -UUSER_DEF1 \
// RUN:   -fdx-record-command-line "clang_dxc -g -Tlib_6_3 -DUSER_DEF0=42 -DUSER_DEF1=43 C:\\\\dx-source-metadata.hlsl" \
// RUN:   -o - %s | FileCheck %s

// RUN: cat %s | %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm \
// RUN:   -debug-info-kind=constructor \
// RUN:   -fdx-record-command-line "clang_dxc -g -Tlib_6_3 C:\\\\dx-source-metadata.hlsl" \
// RUN:   -o - | FileCheck %s --check-prefix=CHECK-FROM-PIPE

// CHECK: !dx.source.contents = !{![[CONTENTS:[0-9]+]]}
// CHECK: !dx.source.defines = !{![[DEFINES:[0-9]+]]}
// CHECK: !dx.source.mainFileName = !{![[MAIN:[0-9]+]]}
// CHECK: !dx.source.args = !{![[ARGS:[0-9]+]]}

// CHECK: ![[CONTENTS]] = !{!"{{.*[\\/]dx-source-metadata.hlsl}}", !"{{.*}}"}
// CHECK: ![[DEFINES]] = !{!"USER_DEF0=42", !"USER_DEF1=43"}
// CHECK: ![[MAIN]] = !{!"{{.*[\\/]dx-source-metadata.hlsl}}"}
// CHECK: ![[ARGS]] = !{!"-g", !"-Tlib_6_3", !"-DUSER_DEF0=42", !"-DUSER_DEF1=43", !"C:\\dx-source-metadata.hlsl"}

// CHECK-FROM-PIPE: !dx.source.contents = !{![[CONTENTS:[0-9]+]]}
// CHECK-FROM-PIPE: !dx.source.mainFileName = !{![[MAIN:[0-9]+]]}
// CHECK-FROM-PIPE: ![[CONTENTS]] = !{!"<stdin>", !"{{.*}}"}
// CHECK-FROM-PIPE: ![[MAIN]] = !{!"<stdin>"}

float foo(float a, float b) {
  return a + b;
}
