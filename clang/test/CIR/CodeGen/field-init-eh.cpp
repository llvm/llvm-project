// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

class Contained {
public:
  Contained(int);
  ~Contained();
};

void mayThrow();

class Container {
public:
  Container() : x(0), contained(1) { mayThrow(); }

  int x;
  Contained contained;
};

void test_field_initializer() {
  Container c;
}

// CIR: cir.func {{.*}} @_ZN9ContainerC2Ev
// CIR:   %[[THIS:.*]] = cir.load %{{.*}}
// CIR:   %[[X_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "x"} : !cir.ptr<!rec_Container> -> !cir.ptr<!s32i>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store align(4) %[[ZERO]], %[[X_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[VOID_PTR_THIS:.*]] = cir.cast bitcast %[[THIS]] : !cir.ptr<!rec_Container> -> !cir.ptr<!u8i>
// CIR:   %[[FOUR:.*]] = cir.const #cir.int<4> : !u64i
// CIR:   %[[CONTAINED_ADDR_VOID:.*]] = cir.ptr_stride %[[VOID_PTR_THIS]], %[[FOUR]] : (!cir.ptr<!u8i>, !u64i) -> !cir.ptr<!u8i>
// CIR:   %[[CONTAINED_ADDR:.*]] = cir.cast bitcast %[[CONTAINED_ADDR_VOID]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_Contained>
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.call @_ZN9ContainedC1Ei(%[[CONTAINED_ADDR]], %[[ONE]])
// CIR:   cir.cleanup.scope {
// CIR:     cir.call @_Z8mayThrowv() : () -> ()
// CIR:     cir.yield
// CIR:   } cleanup eh {
// CIR:     cir.call @_ZN9ContainedD1Ev(%[[CONTAINED_ADDR]])
// CIR:     cir.yield
// CIR:   }

// LLVM: define {{.*}} void @_ZN9ContainerC2Ev
// LLVM:   %[[THIS:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[X_ADDR:.*]] = getelementptr %class.Container, ptr %[[THIS]], i32 0, i32 0
// LLVM:   store i32 0, ptr %[[X_ADDR]]
// LLVM:   %[[CONTAINED_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i64 4
// LLVM:   call void @_ZN9ContainedC1Ei(ptr {{.*}} %[[CONTAINED_ADDR]], i32 {{.*}} 1)
// LLVM:   invoke void @_Z8mayThrowv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:   [[INVOKE_CONT:.*]]:
// LLVM:     br label %[[EXIT:.*]]
// LLVM:   [[LPAD:.*]]:
// LLVM:     %[[EXN:.*]] = landingpad { ptr, i32 }
// LLVM:       cleanup
// LLVM:       br label %[[EH_CLEANUP:.*]]
// LLVM:   [[EH_CLEANUP:.*]]:
// LLVM:     call void @_ZN9ContainedD1Ev(ptr {{.*}} %[[CONTAINED_ADDR]])
// LLVM:     resume { ptr, i32 } %{{.*}}        
// LLVM:   [[EXIT:.*]]:
// LLVM:     ret void

// OGCG: define {{.*}} void @_ZN9ContainerC2Ev
// OGCG:   %[[THIS:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[X_ADDR:.*]] = getelementptr inbounds nuw %class.Container, ptr %[[THIS]], i32 0, i32 0
// OGCG:   store i32 0, ptr %[[X_ADDR]]
// OGCG:   %[[CONTAINED_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 4
// OGCG:   call void @_ZN9ContainedC1Ei(ptr {{.*}} %[[CONTAINED_ADDR]], i32 {{.*}} 1)
// OGCG:   invoke void @_Z8mayThrowv()
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[INVOKE_CONT:.*]]:
// OGCG:   ret void
// OGCG: [[LPAD:.*]]:
// OGCG:   %[[EXN:.*]] = landingpad { ptr, i32 }
// OGCG:                    cleanup
// OGCG:   call void @_ZN9ContainedD1Ev(ptr {{.*}} %[[CONTAINED_ADDR]])
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[EH_RESUME:.*]]:
// OGCG:   resume { ptr, i32 } %{{.*}}
