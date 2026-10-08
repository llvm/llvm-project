// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir -check-prefix=CIR %s
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s


// This is an example right out of the itanium ABI for a 'null' function
// pointer.
struct A { virtual void f(); };
struct B : virtual public A { int i; };
struct C : virtual public A { int j; };
struct D : public B, public C {
  virtual void d();
};
void D::d() {}

// CIR: cir.global {{.*}}@_ZTV1D = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1A1fEv> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1D1dEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 6>, #cir.const_array<[#cir.ptr<-16 : i64> : !cir.ptr<!u8i>, #cir.ptr<-16 : i64> : !cir.ptr<!u8i>, #cir.ptr<-16 : i64> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 5>}>
//
// LLVM: @_ZTV1D = {{.*}}{ [6 x ptr], [5 x ptr] } { [6 x ptr] [ptr null, ptr null, ptr null, ptr @_ZTI1D, ptr @_ZN1A1fEv, ptr @_ZN1D1dEv], [5 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1D, ptr null] }, align 8
//
// CIR: cir.global {{.*}}@_ZTC1D0_1B = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1B> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1A1fEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 5>}>

// LLVM: @_ZTC1D0_1B = {{.*}}{ [5 x ptr] } { [5 x ptr] [ptr null, ptr null, ptr null, ptr @_ZTI1B, ptr @_ZN1A1fEv] }, align 8
//
// CIR: cir.global {{.*}}@_ZTC1D16_1C = #cir.vtable<{#cir.const_array<[#cir.ptr<-16 : i64> : !cir.ptr<!u8i>, #cir.ptr<-16 : i64> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1C> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 5>, #cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<16 : i64> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1C> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1A1fEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>}>
//
// LLVM: @_ZTC1D16_1C = {{.*}}{ [5 x ptr], [4 x ptr] } { [5 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr null, ptr @_ZTI1C, ptr null], [4 x ptr] [ptr null, ptr inttoptr (i64 16 to ptr), ptr @_ZTI1C, ptr @_ZN1A1fEv] }, align 8
