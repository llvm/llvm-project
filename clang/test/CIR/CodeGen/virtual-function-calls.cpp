// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct A {
  A();
  virtual void f(char);
};

// This should initialize the vtable pointer.
A::A() {}

// CIR: !rec_A = !cir.record<struct "A" {!cir.vptr}>
// CIR: !rec_anon_struct = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 3>}>

// CIR: cir.global "private" external @_ZTV1A : !rec_anon_struct

// LLVM: @_ZTV1A = external global { [3 x ptr] }

// OGCG: @_ZTV1A = external unnamed_addr constant { [3 x ptr] }

// CIR: cir.func{{.*}} @_ZN1AC2Ev(%arg0: !cir.ptr<!rec_A> {{.*}})
// CIR:    %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["this", init]
// CIR:    cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CIR:    %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR:    %[[VPTR:.*]] = cir.vtable.address_point(@_ZTV1A, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR:    %[[THIS_VPTR_PTR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
// CIR:    cir.store{{.*}} align(8) %[[VPTR]], %[[THIS_VPTR_PTR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:    cir.return

// LLVM: define{{.*}} void @_ZN1AC2Ev(ptr %[[ARG0:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1A, i64 16), ptr %[[THIS]]

// OGCG: define{{.*}} void @_ZN1AC2Ev(ptr {{.*}} %[[ARG0:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2), ptr %[[THIS]]

// NOTE: The GEP in OGCG looks very different from the one generated with CIR,
//       but it is equivalent. The OGCG GEP indexes by base pointer, then
//       structure, then array, whereas the CIR GEP indexes by byte offset.
