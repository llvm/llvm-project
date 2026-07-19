// 32-bit ARM lowers end-to-end through CIR (GenericARM CXXABI, vtables); records
// and vtables are 4-byte aligned with 4-byte pointers.
//
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -emit-llvm %s -o %t-ogcg.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-ogcg.ll %s

struct S {
  int *p;
  int x;
};

S s;

class A {
public:
  virtual void f();
  int x;
};

void A::f() {}

// CIR-DAG: !rec_S = !cir.struct<"S" {!cir.ptr<!s32i>, !s32i}>
// CIR-DAG: !rec_A = !cir.struct<class "A" {!cir.vptr, !s32i}>
// CIR-DAG: !cir.ptr<!cir.void> = #ptr.spec<size = 32, abi = 32, preferred = 32>
// CIR: cir.global external @s = #cir.zero : !rec_S {alignment = 4 : i64}
// CIR: cir.global {{.*}}@_ZTV1A = #cir.vtable<{{.*}}{alignment = 4 : i64}

// LLVM: @s = global %struct.S zeroinitializer, align 4
// LLVM: @_ZTV1A = global { [3 x ptr] } {{.*}}, align 4

// OGCG: @s = global %struct.S zeroinitializer, align 4
// OGCG: @_ZTV1A = {{.*}}constant { [3 x ptr] } {{.*}}, align 4
