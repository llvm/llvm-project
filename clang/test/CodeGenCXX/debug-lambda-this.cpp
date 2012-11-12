// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -g | FileCheck %s

struct D {
  D();
  D(const D&);
  int x;
  int d(int x);
};
int D::d(int x) {
  [=] {
    return this->x;
  }();
}

// CHECK: metadata !{i32 {{.*}}, metadata !"this", metadata !6, i32 11, i64 64, i64 64, i64 0, i32 1, metadata !37} ; [ DW_TAG_member ] [this] [line 11, size 64, align 64, offset 0] [private] [from ]
