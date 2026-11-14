// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct A {
  virtual void f();
  int a;
};
struct B : virtual A {
  virtual void g();
};
struct C : B {
  virtual void h();
};

void C::h() {}

C c;

// CIR: cir.global constant external  unnamed_addr @_ZTT1C = #cir.const_array<[
// CIR-SAME: #cir.global_view<@_ZTV1C, [0 : i32, 3 : i32], address_point true> : !cir.ptr<!u8i>, 
// CIR-SAME: #cir.global_view<@_ZTC1C0_1B, [0 : i32, 3 : i32], address_point true> : !cir.ptr<!u8i>, 
// CIR-SAME: #cir.global_view<@_ZTC1C0_1B, [1 : i32, 3 : i32], address_point true> : !cir.ptr<!u8i>, 
// CIR-SAME: #cir.global_view<@_ZTV1C, [1 : i32, 3 : i32], address_point true> : !cir.ptr<!u8i>
// CIR-SAME: ]> : !cir.array<!cir.ptr<!u8i> x 4>

// LLVM: @_ZTT1C = {{.*}}constant [4 x ptr] [
// LLVM-SAME: inrange(-12, 8) (i8, ptr @_ZTV1C, i32 12)
// LLVM-SAME: inrange(-12, 4) (i8, ptr @_ZTC1C0_1B, i32 12)
// LLVM-SAME: inrange(-12, 4) (i8, ptr @_ZTC1C0_1B, i32 28)
// LLVM-SAME: inrange(-12, 4) (i8, ptr @_ZTV1C, i32 32)
