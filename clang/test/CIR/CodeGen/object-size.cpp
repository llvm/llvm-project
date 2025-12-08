// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// C++-specific tests for __builtin_object_size

int gi;

// CIR-LABEL: @_Z5test1v
// LLVM-LABEL: define{{.*}} void @_Z5test1v()
// OGCG-LABEL: define{{.*}} void @_Z5test1v()
void test1() {
  // Guaranteeing that our cast removal logic doesn't break more interesting
  // cases.
  struct A { int a; };
  struct B { int b; };
  struct C: public A, public B {};

  C c;

  // CIR: cir.const #cir.int<8>
  // LLVM: store i32 8
  // OGCG: store i32 8
  gi = __builtin_object_size(&c, 0);
  // CIR: cir.const #cir.int<8>
  // LLVM: store i32 8
  // OGCG: store i32 8
  gi = __builtin_object_size((A*)&c, 0);
  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size((B*)&c, 0);

  // CIR: cir.const #cir.int<8>
  // LLVM: store i32 8
  // OGCG: store i32 8
  gi = __builtin_object_size((char*)&c, 0);
  // CIR: cir.const #cir.int<8>
  // LLVM: store i32 8
  // OGCG: store i32 8
  gi = __builtin_object_size((char*)(A*)&c, 0);
  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size((char*)(B*)&c, 0);
}

// CIR-LABEL: @_Z5test2v()
// LLVM-LABEL: define{{.*}} void @_Z5test2v()
// OGCG-LABEL: define{{.*}} void @_Z5test2v()
void test2() {
  struct A { char buf[16]; };
  struct B : A {};
  struct C { int i; B bs[1]; } *c;

  // CIR: cir.objsize max nullunknown %{{.+}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0], 0);
  // CIR: cir.objsize max nullunknown %{{.+}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0], 1);
  // CIR: cir.objsize min nullunknown %{{.+}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0], 2);
  // CIR: cir.const #cir.int<16>
  // LLVM: store i32 16
  // OGCG: store i32 16
  gi = __builtin_object_size(&c->bs[0], 3);

  // NYI: DerivedToBase cast
  // gi = __builtin_object_size((A*)&c->bs[0], 0);

  // CIR: cir.const #cir.int<16>
  // LLVM: store i32 16
  // OGCG: store i32 16
  gi = __builtin_object_size((A*)&c->bs[0], 1);

  // NYI: DerivedToBase cast 
  // gi = __builtin_object_size((A*)&c->bs[0], 2);

  // CIR: cir.const #cir.int<16>
  // LLVM: store i32 16
  // OGCG: store i32 16
  gi = __builtin_object_size((A*)&c->bs[0], 3);

  // CIR: cir.objsize max nullunknown %{{.+}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0].buf[0], 0);
  // CIR: cir.const #cir.int<16>
  // LLVM: store i32 16
  // OGCG: store i32 16
  gi = __builtin_object_size(&c->bs[0].buf[0], 1);
  // CIR: cir.objsize min nullunknown %{{.+}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0].buf[0], 2);
  // CIR: cir.const #cir.int<16>
  // LLVM: store i32 16
  // OGCG: store i32 16
  gi = __builtin_object_size(&c->bs[0].buf[0], 3);
}
