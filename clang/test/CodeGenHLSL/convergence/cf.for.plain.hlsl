// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

int process() {
// CHECK: entry:
// CHECK:   %[[#entry_token:]] = call token @llvm.experimental.convergence.entry()
  int val = 0;

// CHECK: for.cond:
// CHECK-NEXT:   %[[#]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %[[#entry_token]]) ]
// CHECK: br i1 {{.*}}, label %for.body, label %for.end
  for (int i = 0; i < 10; ++i) {

// CHECK: for.body:
// CHECK:   br label %for.inc
    val = i;

// CHECK: for.inc:
// CHECK:   br label %for.cond
  }

// CHECK: for.end:
// CHECK:   br label %for.cond1

  // Infinite loop
  for ( ; ; ) {
// CHECK: for.cond1:
// CHECK-NEXT:   %[[#]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %[[#entry_token]]) ]
// CHECK:   br label %for.cond1
    val = 0;
  }

// CHECK-NEXT: }
// This loop in unreachable. Not generated.
  // Null body
  for (int j = 0; j < 10; ++j)
  ;
  return val;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
