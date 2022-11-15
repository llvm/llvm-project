// REQUIRES: amdgpu-registered-target
// RUN: %clang -g -target amdgcn-amd-amdhsa -march=gfx900 -O0 -nogpulib %s -c -o - | llvm-dwarfdump -v -debug-info - | FileCheck "%s"
// CHECK: DW_TAG_subprogram
// CHECK: DW_AT_name [DW_FORM_strx1]
// CHECK-SAME: (indexed ({{[0-9a-f]+}}) string = "foo")
//
// CHECK: DW_TAG_formal_parameter
// CHECK: DW_AT_name [DW_FORM_strx1]
// CHECK-SAME: (indexed ({{[0-9a-f]+}}) string = "data")
// CHECK: DW_AT_type [DW_FORM_ref4]
// CHECK-SAME: (cu + 0x{{[0-9a-f]+}} => {0x[[BAR_OFFSET:[0-9a-f]+]]} "bar")
//
// CHECK: DW_TAG_variable
// CHECK: DW_AT_name [DW_FORM_strx1]
// CHECK-SAME: (indexed ({{[0-9a-f]+}}) string = "offset")
//
// CHECK: 0x[[BAR_OFFSET]]: DW_TAG_structure_type
// CHECK: DW_AT_name [DW_FORM_strx1]
// CHECK-SAME: (indexed ({{[0-9a-f]+}}) string = "bar")
//
// CHECK: DW_TAG_member
// CHECK: DW_AT_name [DW_FORM_strx1]
// CHECK-SAME: (indexed ({{[0-9a-f]+}}) string = "C")
//
// CHECK: DW_TAG_member
// CHECK: DW_AT_name [DW_FORM_strx1]
// CHECK-SAME: (indexed ({{[0-9a-f]+}}) string = "A")
struct bar {
  __global unsigned *C;
  __global unsigned *A;
};

void foo(struct bar data) {
  unsigned offset = get_global_id(0);
  data.C[offset] = data.A[offset];
}
