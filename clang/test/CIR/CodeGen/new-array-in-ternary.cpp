// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fcxx-exceptions -fexceptions %s -o %t-eh.cir
// RUN: FileCheck --input-file=%t-eh.cir %s --check-prefix=CIR-EH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fcxx-exceptions -fexceptions %s -o %t-cir-eh.ll
// RUN: FileCheck --input-file=%t-cir-eh.ll %s --check-prefixes=LLVM-EH,LLVMCIR-EH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fcxx-exceptions -fexceptions %s -o %t-eh.ll
// RUN: FileCheck --input-file=%t-eh.ll %s --check-prefixes=LLVM-EH,OGCG-EH

struct APFloat {
  struct Storage {
    ~Storage();
  } U;
};

struct DoubleAPFloat {
  APFloat *Floats;
  DoubleAPFloat();
};

DoubleAPFloat::DoubleAPFloat()
    : Floats(Floats ? new APFloat[]{} : nullptr) {}

// CIR-LABEL: cir.func {{.*}} @_ZN13DoubleAPFloatC2Ev
// CIR:         %[[RESULT:.*]] = cir.ternary({{.*}}, true {
// CIR:           %[[ALLOC:.*]] = cir.call @_Znam
// CIR:           %[[NEW_PTR:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!{{void|cir.void}}> -> !cir.ptr<!rec_APFloat>
// CIR:           cir.yield %{{.*}} : !cir.ptr<!rec_APFloat>
// CIR:         }, false {
// CIR:           %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_APFloat>
// CIR:           cir.yield %[[NULL]] : !cir.ptr<!rec_APFloat>
// CIR:         }) : (!cir.bool) -> !cir.ptr<!rec_APFloat>
// CIR:         cir.store {{.*}} %[[RESULT]], %{{.*}} : !cir.ptr<!rec_APFloat>, !cir.ptr<!cir.ptr<!rec_APFloat>>

// LLVM-LABEL: define dso_local void @_ZN13DoubleAPFloatC2Ev(
// LLVM:         %[[CMP:.*]] = icmp ne ptr %{{.*}}, null
// LLVM:         br i1 %[[CMP]], label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// LLVM:       [[TRUE_BR]]:
// LLVM:         %[[ALLOC:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} 8)
// LLVM:         %[[COOKIE_END:.*]] = getelementptr {{(inbounds )?}}i8, ptr %[[ALLOC]], i64 8
// LLVM:         br label %{{.*}}
// LLVM:       [[FALSE_BR]]:
// LLVM:         br label %{{.*}}
// LLVMCIR:      %[[PHI:.*]] = phi ptr [ null, %{{.*}} ], [ %{{.*}}, %{{.*}} ]
// OGCG:         %[[PHI:.*]] = phi ptr [ %[[COOKIE_END]], %[[TRUE_BR]] ], [ null, %[[FALSE_BR]] ]
// LLVM:         store ptr %[[PHI]], ptr %{{.*}}
// LLVM:         ret void

// CIR-EH-LABEL: cir.func {{.*}} @_ZN13DoubleAPFloatC2Ev
// CIR-EH:         %[[ACTIVE:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CIR-EH:         %[[FALSE_INIT:.*]] = cir.const #false
// CIR-EH:         cir.store %[[FALSE_INIT]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-EH:         %[[RESULT:.*]] = cir.ternary({{.*}}, true {
// CIR-EH:           %[[NEW_RESULT:.*]] = cir.alloca "__new_result" {{.*}} : !cir.ptr<!cir.ptr<!rec_APFloat>>
// CIR-EH:           %[[SPILL:.*]] = cir.alloca "tmp.exprcleanup" {{.*}} : !cir.ptr<!cir.ptr<!rec_APFloat>>
// CIR-EH:           cir.call @_Znam
// CIR-EH:           cir.cleanup.scope {
// CIR-EH:             %[[TRUE:.*]] = cir.const #true
// CIR-EH:             cir.store %[[TRUE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-EH:             cir.cleanup.scope {
// CIR-EH:               %[[FALSE:.*]] = cir.const #false
// CIR-EH:               cir.store %[[FALSE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-EH:               cir.store {{.*}} %{{.*}}, %[[SPILL]] : !cir.ptr<!rec_APFloat>, !cir.ptr<!cir.ptr<!rec_APFloat>>
// CIR-EH:               cir.yield
// CIR-EH:             } cleanup eh {
// CIR-EH:               cir.call @_ZN7APFloatD1Ev
// CIR-EH:             }
// CIR-EH:             cir.yield
// CIR-EH:           } cleanup eh {
// CIR-EH:             %[[IS_ACTIVE:.*]] = cir.load {{.*}} %[[ACTIVE]]
// CIR-EH:             cir.if %[[IS_ACTIVE]] {
// CIR-EH:               cir.call @_ZdaPvm
// CIR-EH:             }
// CIR-EH:           }
// CIR-EH:           %[[RELOAD:.*]] = cir.load {{.*}} %[[SPILL]] : !cir.ptr<!cir.ptr<!rec_APFloat>>, !cir.ptr<!rec_APFloat>
// CIR-EH:           cir.yield %[[RELOAD]] : !cir.ptr<!rec_APFloat>
// CIR-EH:         }, false {
// CIR-EH:           %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_APFloat>
// CIR-EH:           cir.yield %[[NULL]] : !cir.ptr<!rec_APFloat>
// CIR-EH:         }) : (!cir.bool) -> !cir.ptr<!rec_APFloat>
// CIR-EH:         cir.store {{.*}} %[[RESULT]], %{{.*}} : !cir.ptr<!rec_APFloat>, !cir.ptr<!cir.ptr<!rec_APFloat>>


// LLVM-EH-LABEL: define dso_local void @_ZN13DoubleAPFloatC2Ev(
// LLVMCIR-EH:      %[[ACTIVE:.*]] = alloca i8
// OGCG-EH:         %[[ACTIVE:.*]] = alloca i1
// LLVM-EH:         %[[CMP:.*]] = icmp ne ptr %{{.*}}, null
// LLVMCIR-EH:      store i8 0, ptr %[[ACTIVE]]
// OGCG-EH:         store i1 false, ptr %[[ACTIVE]]
// LLVM-EH:         br i1 %[[CMP]], label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// LLVM-EH:       [[TRUE_BR]]:
// LLVM-EH:         %[[ALLOC:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} 8)
// LLVMCIR-EH:      store i8 1, ptr %[[ACTIVE]]
// OGCG-EH:         store i1 true, ptr %[[ACTIVE]]
// LLVMCIR-EH:      store i8 0, ptr %[[ACTIVE]]
// LLVM-EH:         br label %{{.*}}
// LLVM-EH:       [[FALSE_BR]]:
// LLVM-EH:         br label %{{.*}}
// LLVM-EH:         %[[PHI:.*]] = phi ptr
// LLVM-EH:         store ptr %[[PHI]], ptr %{{.*}}
// LLVM-EH:         ret void

struct T {
  ~T();
};
extern T globalT;
T &pickRef(bool b) {
  return b ? *new T[]{T{}, T{}} : globalT;
}

// CIR-LABEL: cir.func {{.*}} @_Z7pickRefb
// CIR:         %[[RES:.*]] = cir.ternary({{.*}}, true {
// CIR:           cir.call @_Znam
// CIR-NOT:       cir.cleanup.scope
// CIR:           cir.yield %{{.*}} : !cir.ptr<!rec_T>
// CIR:         }, false {
// CIR:           %[[GREF:.*]] = cir.get_global @globalT : !cir.ptr<!rec_T>
// CIR:           cir.yield %[[GREF]] : !cir.ptr<!rec_T>
// CIR:         }) : (!cir.bool) -> !cir.ptr<!rec_T>

// LLVM-LABEL: define dso_local {{.*}}ptr @_Z7pickRefb(
// LLVMCIR:      %[[CMP:.*]] = trunc i8 %{{.*}} to i1
// OGCG:         %[[CMP:.*]] = icmp ne i8 %{{.*}}, 0
// LLVM:         br i1 %[[CMP]], label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// LLVM:       [[TRUE_BR]]:
// LLVM:         %[[ALLOC:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} 10)
// LLVM:         %[[COOKIE_END:.*]] = getelementptr {{(inbounds )?}}i8, ptr %[[ALLOC]], i64 8
// LLVM:         br label %{{.*}}
// LLVM:       [[FALSE_BR]]:
// LLVM:         br label %{{.*}}
// LLVMCIR:      %[[PHI:.*]] = phi ptr [ @globalT, %{{.*}} ], [ %{{.*}}, %{{.*}} ]
// OGCG:         %[[PHI:.*]] = phi ptr [ %[[COOKIE_END]], %[[TRUE_BR]] ], [ @globalT, %[[FALSE_BR]] ]
// LLVM:         ret ptr %{{.*}}

// CIR-EH-LABEL: cir.func {{.*}} @_Z7pickRefb
// CIR-EH:         %[[RES:.*]] = cir.ternary({{.*}}, true {
// CIR-EH:           %[[SPILL:.*]] = cir.alloca "tmp.exprcleanup" {{.*}} : !cir.ptr<!cir.ptr<!rec_T>>
// CIR-EH:           cir.call @_Znam
// CIR-EH:           cir.cleanup.scope {
// CIR-EH:             cir.cleanup.scope {
// CIR-EH:               cir.store {{.*}} %{{.*}}, %[[SPILL]] : !cir.ptr<!rec_T>, !cir.ptr<!cir.ptr<!rec_T>>
// CIR-EH:               cir.yield
// CIR-EH:             } cleanup eh {
// CIR-EH:               cir.call @_ZN1TD1Ev
// CIR-EH:             }
// CIR-EH:             cir.yield
// CIR-EH:           } cleanup eh {
// CIR-EH:             cir.call @_ZdaPvm
// CIR-EH:           }
// CIR-EH:           %[[RELOAD:.*]] = cir.load {{.*}} %[[SPILL]] : !cir.ptr<!cir.ptr<!rec_T>>, !cir.ptr<!rec_T>
// CIR-EH:           cir.yield %[[RELOAD]] : !cir.ptr<!rec_T>
// CIR-EH:         }, false {
// CIR-EH:           %[[GREF:.*]] = cir.get_global @globalT : !cir.ptr<!rec_T>
// CIR-EH:           cir.yield %[[GREF]] : !cir.ptr<!rec_T>
// CIR-EH:         }) : (!cir.bool) -> !cir.ptr<!rec_T>

// LLVM-EH-LABEL: define dso_local {{.*}}ptr @_Z7pickRefb(
// LLVMCIR-EH:      alloca i8
// LLVMCIR-EH:      %[[ACTIVE:.*]] = alloca i8
// OGCG-EH:         %[[ACTIVE:.*]] = alloca i1
// LLVMCIR-EH:      %[[CMP:.*]] = trunc i8 %{{.*}} to i1
// OGCG-EH:         %[[CMP:.*]] = icmp ne i8 %{{.*}}, 0
// LLVMCIR-EH:      store i8 0, ptr %[[ACTIVE]]
// OGCG-EH:         store i1 false, ptr %[[ACTIVE]]
// LLVM-EH:         br i1 %[[CMP]], label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// LLVM-EH:       [[TRUE_BR]]:
// LLVM-EH:         %[[ALLOC:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} 10)
// LLVMCIR-EH:      store i8 1, ptr %[[ACTIVE]]
// OGCG-EH:         store i1 true, ptr %[[ACTIVE]]
// LLVMCIR-EH:      store i8 0, ptr %[[ACTIVE]]
// LLVM-EH:         br label %{{.*}}
// LLVM-EH:       [[FALSE_BR]]:
// LLVM-EH:         br label %{{.*}}
// LLVMCIR-EH:      %[[PHI:.*]] = phi ptr [ @globalT, %{{.*}} ], [ %{{.*}}, %{{.*}} ]
// OGCG-EH:         %[[PHI:.*]] = phi ptr [ %{{.*}}, %[[TRUE_BR]] ], [ @globalT, %[[FALSE_BR]] ]
// LLVM-EH:         ret ptr %{{.*}}
