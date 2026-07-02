// RUN: %clang_cc1 -std=c++20 -triple nvptx-nvidia-cuda -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple nvptx-nvidia-cuda -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple nvptx-nvidia-cuda -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// On a target with 32-bit pointers (e.g. nvptx) both a data pointer (!cir.ptr)
// and the vtable pointer (!cir.vptr) are 4 bytes wide. The pointer width is
// carried by a #ptr.spec data-layout entry keyed on cir.ptr, so the field
// following a pointer lands at the AST-mandated offset. Sizing pointers as a
// hardcoded 64 bits previously tripped the record layout builder (insertPadding:
// assertion `offset >= size`) on every record containing a pointer.

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

// The module carries a #ptr.spec pointer data-layout entry (size/abi/preferred
// in bits) that drives both cir.ptr and cir.vptr widths. The 4-byte pointer is
// immediately followed by 'x' at offset 4 with no padding, and each record is
// 4-byte aligned.
// CIR-DAG: !rec_S = !cir.struct<"S" {!cir.ptr<!s32i>, !s32i}>
// CIR-DAG: !rec_A = !cir.struct<class "A" {!cir.vptr, !s32i}>
// CIR-DAG: !cir.ptr<!cir.void> = #ptr.spec<size = 32, abi = 32, preferred = 32>
// CIR: cir.global external @s = #cir.zero : !rec_S {alignment = 4 : i64}
// CIR: cir.global{{.*}}@_ZTV1A = #cir.vtable<{{.*}}{alignment = 4 : i64}

// LLVM: @s = global %struct.S zeroinitializer, align 4
// LLVM: @_ZTV1A = global { [3 x ptr] } {{.*}}, align 4

// OGCG: @s = global %struct.S zeroinitializer, align 4
// OGCG: @_ZTV1A = {{.*}}constant { [3 x ptr] } {{.*}}, align 4
