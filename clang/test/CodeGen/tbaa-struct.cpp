// RUN: %clang_cc1 -emit-llvm -o - -O1 %s | FileCheck %s
//
// Check that we generate !tbaa.struct metadata for struct copies.
struct A {
  short s;
  int i;
  char c;
  int j;
};

void copy(struct A *a, struct A *b) {
  *a = *b;
}

// CHECK: target datalayout = "{{.*}}p:[[P:64|32]]
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i[[P]](i8* %{{.*}}, i8* %{{.*}}, i[[P]] 16, i32 4, i1 false), !tbaa.struct [[TS:!.*]]
// CHECK: [[TS]] = metadata !{i64 0, i64 2, metadata !{{.*}}, i64 4, i64 4, metadata !{{.*}}, i64 8, i64 1, metadata !{{.*}}, i64 12, i64 4, metadata !{{.*}}}
