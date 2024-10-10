// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: define spir_func void @{{.*main.*}}() [[A0:#[0-9]+]] {
void main() {
// CHECK: entry:
// CHECK:   %[[CT_ENTRY:[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK:   br label %[[LABEL_WHILE_COND:.+]]
  int cond = 0;

// CHECK: [[LABEL_WHILE_COND]]:
// CHECK:   %[[CT_LOOP:[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %[[CT_ENTRY]]) ]
// CHECK:   br label %[[LABEL_WHILE_BODY:.+]]
  while (true) {

// CHECK: [[LABEL_WHILE_BODY]]:
// CHECK:   br i1 {{%.+}}, label %[[LABEL_IF_THEN:.+]], label %[[LABEL_IF_END:.+]]

// CHECK: [[LABEL_IF_THEN]]:
// CHECK:   call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %[[CT_LOOP]]) ]
// CHECK:   br label %[[LABEL_WHILE_END:.+]]
    if (cond == 2) {
      uint index = WaveGetLaneIndex();
      break;
    }

// CHECK: [[LABEL_IF_END]]:
// CHECK:   br label %[[LABEL_WHILE_COND]]
    cond++;
  }

// CHECK: [[LABEL_WHILE_END]]:
// CHECK:   ret void
}

// CHECK-DAG: declare i32 @__hlsl_wave_get_lane_index() [[A1:#[0-9]+]]

// CHECK-DAG: attributes [[A0]] = {{{.*}}convergent{{.*}}}
// CHECK-DAG: attributes [[A1]] = {{{.*}}convergent{{.*}}}

