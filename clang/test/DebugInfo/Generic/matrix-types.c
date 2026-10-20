// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -debug-info-kind=limited -emit-llvm -disable-llvm-passes -o - | FileCheck %s

typedef double dx2x3_t __attribute__((matrix_type(2, 3)));

void load_store_double(dx2x3_t *a, dx2x3_t *b) {
  // CHECK-DAG:  #dbg_declare(ptr %a.addr, [[EXPR_A:![0-9]+]]
  // CHECK-DAG:  #dbg_declare(ptr %b.addr, [[EXPR_B:![0-9]+]]
  // CHECK: [[PTR_TY:![0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[TYPEDEF:![0-9]+]], size: 64)
  // CHECK: [[TYPEDEF]] = !DIDerivedType(tag: DW_TAG_typedef, name: "dx2x3_t", {{.+}} baseType: [[MATRIX_TY:![0-9]+]])
  // CHECK: [[MATRIX_TY]] = !DICompositeType(tag: DW_TAG_array_type, baseType: [[ELT_TY:![0-9]+]], size: 384, elements: [[ELEMENTS:![0-9]+]])
  // CHECK: [[ELT_TY]] = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
  // CHECK: [[ELEMENTS]] = !{[[COLS:![0-9]+]], [[ROWS:![0-9]+]]}
  // CHECK: [[COLS]] = !DISubrange(count: 3)
  // CHECK: [[ROWS]] = !DISubrange(count: 2)
  // CHECK: [[EXPR_A]] = !DILocalVariable(name: "a", arg: 1, {{.+}} type: [[PTR_TY]])
  // CHECK: [[EXPR_B]] = !DILocalVariable(name: "b", arg: 2, {{.+}} type: [[PTR_TY]])

  *a = *b;
}
