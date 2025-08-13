// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

struct A {
  A();
  virtual void f(char);
};

// This should initialize the vtable pointer.
A::A() {}

// CIR: !rec_A = !cir.record<struct "A" {!cir.vptr}>
// CIR: !rec_anon_struct = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 3>}>

// CIR: cir.global "private" external @_ZTV1A : !rec_anon_struct

// CIR: cir.func dso_local @_ZN1AC2Ev(%arg0: !cir.ptr<!rec_A> {{.*}})
// CIR:    %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["this", init]
// CIR:    cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CIR:    %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR:    %[[VPTR:.*]] = cir.vtable.address_point(@_ZTV1A, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR:    %[[THIS_VPTR_PTR:.*]] = cir.cast(bitcast, %[[THIS]] : !cir.ptr<!rec_A>), !cir.ptr<!cir.vptr>
// CIR:    cir.store align(8) %[[VPTR]], %[[THIS_VPTR_PTR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:    cir.return
