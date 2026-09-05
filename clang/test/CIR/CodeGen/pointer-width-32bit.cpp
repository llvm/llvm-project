// RUN: %clang_cc1 -std=c++20 -triple nvptx-nvidia-cuda -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple nvptx-nvidia-cuda -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple nvptx-nvidia-cuda -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// On a 32-bit-pointer target such as nvptx, !cir.ptr and !cir.vptr are 4 bytes
// wide, driven by the #cir.ptr_spec data-layout entry. Hardcoded 64-bit widths
// used to trip the record layout builder (insertPadding: offset >= size) on
// every record containing a pointer.

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

// A data member pointer is ptrdiff_t-sized, so 'x' again lands at offset 4.
struct M {
  int S::*pm;
  int x;
};

M m;

// Each 4-byte pointer is followed by 'x' at offset 4 with no padding; records
// are 4-byte aligned.
// CIR-DAG: !rec_S = !cir.struct<"S" {data !cir.ptr<!s32i>, data !s32i}>
// CIR-DAG: !rec_A = !cir.struct<class "A" {data !cir.vptr, data !s32i}>
// -emit-cir prints after cir-cxxabi-lowering, so M's member pointer is
// already a 32-bit integer here.
// CIR-DAG: !rec_M = !cir.struct<"M" {data !s32i, data !s32i}>
// CIR-DAG: !cir.ptr<!cir.void> = #cir.ptr_spec<size = 32, abi = 32, preferred = 32, index = 32>
// CIR: cir.global external @s = #cir.zero : !rec_S {alignment = 4 : i64}
// CIR: cir.global external @m = #cir.const_record<{#cir.int<-1> : !s32i, #cir.int<0> : !s32i}> : !rec_M {alignment = 4 : i64}
// CIR: cir.global{{.*}}@_ZTV1A = #cir.vtable<{{.*}}{alignment = 4 : i64}

// LLVM: @s = global %struct.S zeroinitializer, align 4
// LLVM: @m = global %struct.M { i32 -1, i32 0 }, align 4
// LLVM: @_ZTV1A = constant { [3 x ptr] } {{.*}}, align 4

// OGCG: @s = global %struct.S zeroinitializer, align 4
// OGCG: @m = global %struct.M { i32 -1, i32 0 }, align 4
// OGCG: @_ZTV1A = {{.*}}constant { [3 x ptr] } {{.*}}, align 4
