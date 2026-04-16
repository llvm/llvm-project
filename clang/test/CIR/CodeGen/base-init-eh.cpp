// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

class Base {
public:
  Base(int);
  ~Base();
};

void mayThrow();

class Derived : public Base {
public:
  Derived() : Base(0) { mayThrow(); }
};

void test_base_initializer() {
  Derived d;
}

// CIR: cir.func {{.*}} @_ZN7DerivedC2Ev
// CIR:   %[[THIS:.*]] = cir.load %{{.*}}
// CIR:   %[[BASE_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.call @_ZN4BaseC2Ei(%[[BASE_ADDR]], %[[ZERO]])
// CIR:   cir.cleanup.scope {
// CIR:     cir.call @_Z8mayThrowv() : () -> ()
// CIR:     cir.yield
// CIR:   } cleanup eh {
// CIR:     cir.call @_ZN4BaseD2Ev(%[[BASE_ADDR]])
// CIR:     cir.yield
// CIR:   }

// LLVM: define {{.*}} void @_ZN7DerivedC2Ev
// LLVM:   %[[THIS:.*]] = load ptr, ptr %{{.*}}
// LLVM:   call void @_ZN4BaseC2Ei(ptr {{.*}} %[[THIS]], i32 {{.*}} 0)
// LLVM:   invoke void @_Z8mayThrowv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:   [[INVOKE_CONT:.*]]:
// LLVM:     br label %[[EXIT:.*]]
// LLVM:   [[LPAD:.*]]:
// LLVM:     %[[EXN:.*]] = landingpad { ptr, i32 }
// LLVM:       cleanup
// LLVM:       br label %[[EH_CLEANUP:.*]]
// LLVM:   [[EH_CLEANUP:.*]]:
// LLVM:     call void @_ZN4BaseD2Ev(ptr {{.*}} %[[THIS]])
// LLVM:     resume { ptr, i32 } %{{.*}}        
// LLVM:   [[EXIT:.*]]:
// LLVM:     ret void

// OGCG emits @_ZN7DerivedC2Ev below @_ZN11VirtDerivedC1Ev

class VirtDerived : public virtual Base {
public:
  VirtDerived() : Base(0) { mayThrow(); }
};

void test_virt_base_initializer() {
  VirtDerived v;
}
        
// CIR: cir.func {{.*}} @_ZN11VirtDerivedC1Ev
// CIR:   %[[THIS:.*]] = cir.load %{{.*}}
// CIR:   %[[BASE_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_VirtDerived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.call @_ZN4BaseC2Ei(%[[BASE_ADDR]], %[[ZERO]])
// CIR:   cir.cleanup.scope {
// CIR:     %[[VTABLE_ADDR:.*]] = cir.vtable.address_point(@_ZTV11VirtDerived, address_point = <index = 0, offset = 3>) : !cir.vptr
// CIR:     %[[VTABLE_PTR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_VirtDerived> -> !cir.ptr<!cir.vptr>
// CIR:     cir.store{{.*}} %[[VTABLE_ADDR]], %[[VTABLE_PTR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:     cir.call @_Z8mayThrowv() : () -> ()
// CIR:     cir.yield
// CIR:   } cleanup eh {
// CIR:     cir.call @_ZN4BaseD2Ev(%[[BASE_ADDR]])
// CIR:     cir.yield
// CIR:   }

// LLVM: define {{.*}} void @_ZN11VirtDerivedC1Ev
// LLVM:   %[[THIS:.*]] = load ptr, ptr %{{.*}}
// LLVM:   call void @_ZN4BaseC2Ei(ptr {{.*}} %[[THIS]], i32 {{.*}} 0)
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV11VirtDerived, i64 24), ptr %[[THIS]]
// LLVM:   invoke void @_Z8mayThrowv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:   [[INVOKE_CONT:.*]]:
// LLVM:     br label %[[EXIT:.*]]
// LLVM:   [[LPAD:.*]]:
// LLVM:     %[[EXN:.*]] = landingpad { ptr, i32 }
// LLVM:       cleanup
// LLVM:       br label %[[EH_CLEANUP:.*]]
// LLVM:   [[EH_CLEANUP:.*]]:
// LLVM:     call void @_ZN4BaseD2Ev(ptr {{.*}} %[[THIS]])
// LLVM:     resume { ptr, i32 } %{{.*}}
// LLVM:   [[EXIT:.*]]:
// LLVM:     ret void

// OGCG: define {{.*}} void @_ZN11VirtDerivedC1Ev
// OGCG:   %[[THIS:.*]] = load ptr, ptr %{{.*}}
// OGCG:   call void @_ZN4BaseC2Ei(ptr {{.*}} %[[THIS]], i32 {{.*}} 0)
// OGCG:   store ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr] }, ptr @_ZTV11VirtDerived, i32 0, i32 0, i32 3), ptr %[[THIS]]
// OGCG:   invoke void @_Z8mayThrowv()
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[INVOKE_CONT:.*]]:
// OGCG:   ret void
// OGCG: [[LPAD:.*]]:
// OGCG:   %[[EXN:.*]] = landingpad { ptr, i32 }
// OGCG:                    cleanup
// OGCG:   call void @_ZN4BaseD2Ev(ptr {{.*}} %[[THIS]])
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[EH_RESUME:.*]]:
// OGCG:   resume { ptr, i32 } %{{.*}}

// OGCG: define {{.*}} void @_ZN7DerivedC2Ev
// OGCG:   %[[THIS:.*]] = load ptr, ptr %{{.*}}
// OGCG:   call void @_ZN4BaseC2Ei(ptr {{.*}} %[[THIS]], i32 {{.*}} 0)
// OGCG:   invoke void @_Z8mayThrowv()
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[INVOKE_CONT:.*]]:
// OGCG:   ret void
// OGCG: [[LPAD:.*]]:
// OGCG:   %[[EXN:.*]] = landingpad { ptr, i32 }
// OGCG:                    cleanup
// OGCG:   call void @_ZN4BaseD2Ev(ptr {{.*}} %[[THIS]])
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[EH_RESUME:.*]]:
// OGCG:   resume { ptr, i32 } %{{.*}}
