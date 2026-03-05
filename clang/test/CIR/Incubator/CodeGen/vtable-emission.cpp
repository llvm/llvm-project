// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir  -emit-llvm -o - %s \
// RUN: | opt -S -passes=instcombine,mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct S {
  virtual void key();
  virtual void nonKey() {}
} sobj;

void S::key() {}

// CHECK-DAG: !rec_anon_struct1 = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 4>}>
// CHECK-DAG: !rec_anon_struct2 = !cir.record<struct  {!cir.ptr<!rec_anon_struct1>}>

// The definition of the key function should result in the vtable being emitted.
// CHECK: cir.global constant external @_ZTV1S = #cir.vtable
// LLVM: @_ZTV1S = constant { [4 x ptr] } { [4 x ptr]
// LLVM-SAME: [ptr null, ptr @_ZTI1S, ptr @_ZN1S3keyEv, ptr @_ZN1S6nonKeyEv] }, align 8

// Note: GEP emitted by cir might not be the same as LLVM, due to constant folding.
// CHECK: cir.global external @sobj = #cir.const_record
// CHECK-SAME: <{#cir.global_view<@_ZTV1S, [0 : i32, 2 : i32]> :
// CHECK-SAME: !cir.ptr<!rec_anon_struct1>}> : !rec_anon_struct2 {alignment = 8 : i64}
// LLVM: @sobj = global { ptr } { ptr getelementptr inbounds nuw
// LLVM-SAME: (i8, ptr @_ZTV1S, i64 16) }, align 8

// The reference from the vtable should result in nonKey being emitted.
// CHECK: cir.func {{.*}} @_ZN1S6nonKeyEv({{.*}} {
