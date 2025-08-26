// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir  -emit-llvm -o %t-cir.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Note: This test is using -fno-rtti so that we can delay implemntation of that handling.
//       When rtti handling for vtables is implemented, that option should be removed.

struct S {
  virtual void key();
  virtual void nonKey() {}
};

void S::key() {}

// CHECK-DAG: !rec_anon_struct = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 4>}>

// The definition of the key function should result in the vtable being emitted.
// CHECK:      cir.global "private" external @_ZTV1S = #cir.vtable<{
// CHECK-SAME:     #cir.const_array<[
// CHECK-SAME:         #cir.ptr<null> : !cir.ptr<!u8i>,
// CHECK-SAME:         #cir.ptr<null> : !cir.ptr<!u8i>,
// CHECK-SAME:         #cir.global_view<@_ZN1S3keyEv> : !cir.ptr<!u8i>,
// CHECK-SAME:         #cir.global_view<@_ZN1S6nonKeyEv> : !cir.ptr<!u8i>]>
// CHECK-SAME:     : !cir.array<!cir.ptr<!u8i> x 4>}> : !rec_anon_struct

// LLVM:      @_ZTV1S = global { [4 x ptr] } { [4 x ptr]
// LLVM-SAME:      [ptr null, ptr null, ptr @_ZN1S3keyEv, ptr @_ZN1S6nonKeyEv] }

// OGCG:      @_ZTV1S = unnamed_addr constant { [4 x ptr] } { [4 x ptr]
// OGCG-SAME:      [ptr null, ptr null, ptr @_ZN1S3keyEv, ptr @_ZN1S6nonKeyEv] }

// CHECK: cir.func dso_local @_ZN1S3keyEv

// The reference from the vtable should result in nonKey being emitted.
// CHECK: cir.func comdat linkonce_odr @_ZN1S6nonKeyEv
