// RUN: %clang_cc1 -emit-llvm -std=c++11 -debug-info-kind=limited %s -o -| FileCheck %s

void foo() {
  decltype(nullptr) t = 0;
}

// CHECK: !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
