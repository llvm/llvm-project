// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fexceptions \
// RUN:     -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fexceptions \
// RUN:     -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fexceptions \
// RUN:     -fcxx-exceptions %s -emit-llvm -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct Throws {
  Throws(int);
  Throws();
  ~Throws();
};

// Constant-bound array new with a braced initializer list of the same length.
// There is no constructor loop because the bound matches the number of
// initialisers, but the partial-destruction cleanup for the init-list portion
// still has to be emitted because Throws has a non-trivial destructor.

void cleanup_const_exact() { new Throws[3]{1, 2, 3}; }

// In CIR, the irregular partial-array EH cleanup is represented by a
// cir.cleanup.scope around the three constructor calls whose EH region
// destroys [BeginPtr, *arrayinit.endOfInit) in reverse. Before each ctor
// call we update arrayinit.endOfInit so the EH cleanup destroys only the
// elements that were actually constructed before the throw.
// CIR-LABEL: cir.func{{.*}} @_Z19cleanup_const_exactv(
// CIR:   %[[INIT_END:.*]] = cir.alloca {{.*}}, !cir.ptr<!cir.ptr<!rec_Throws>>, ["arrayinit.endOfInit"]
// CIR:   %[[ELT0:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!void> -> !cir.ptr<!rec_Throws>
// CIR:   cir.cleanup.scope {
// CIR:     cir.store align(8) %[[ELT0]], %[[INIT_END]]
// CIR:     cir.call @_ZN6ThrowsC1Ei(%[[ELT0]], %{{.*}})
// CIR:     %[[ELT1:.*]] = cir.ptr_stride %[[ELT0]], %{{.*}}
// CIR:     cir.store align(8) %[[ELT1]], %[[INIT_END]]
// CIR:     cir.call @_ZN6ThrowsC1Ei(%[[ELT1]], %{{.*}})
// CIR:     %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %{{.*}}
// CIR:     cir.store align(8) %[[ELT2]], %[[INIT_END]]
// CIR:     cir.call @_ZN6ThrowsC1Ei(%[[ELT2]], %{{.*}})
// CIR:   } cleanup eh {
// CIR:     %[[END:.*]] = cir.load align(8) %[[INIT_END]]
// CIR:     %[[NE:.*]] = cir.cmp ne %[[END]], %[[ELT0]]
// CIR:     cir.if %[[NE]] {
// CIR:       cir.do {
// CIR:         %{{.*}} = cir.ptr_stride %{{.*}}, %{{.*}} : (!cir.ptr<!rec_Throws>, !s64i) -> !cir.ptr<!rec_Throws>
// CIR:         cir.call @_ZN6ThrowsD1Ev(%{{.*}}) nothrow
// CIR:       } while {
// CIR:         cir.cmp ne %{{.*}}, %[[ELT0]]
// CIR:       }
// CIR:     }
// CIR:   }
// CIR:   cir.call @_ZdaPv(

// CIR-emitted LLVM lowers the same logic to a single landing pad shared by
// all three init-list invokes; that landing pad runs the reverse-destruction
// loop using arrayinit.endOfInit as the upper bound and then operator delete[].
// LLVM-LABEL: define{{.*}} void @_Z19cleanup_const_exactv(
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:   %[[ALLOC:.*]] = call{{.*}}ptr @_Znam(i64 noundef 11)
// LLVM:   store i64 3, ptr %[[ALLOC]]
// LLVM:   %[[FIRST:.*]] = getelementptr i8, ptr %[[ALLOC]], i64 8
// LLVM:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[FIRST]], i32 noundef 1)
// LLVM-NEXT: to label %{{.*}} unwind label %[[LPAD:.*]]
// LLVM:   %[[ELT1:.*]] = getelementptr %struct.Throws, ptr %[[FIRST]], i64 1
// LLVM:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[ELT1]], i32 noundef 2)
// LLVM-NEXT: to label %{{.*}} unwind label %[[LPAD]]
// LLVM:   %[[ELT2:.*]] = getelementptr %struct.Throws, ptr %[[ELT1]], i64 1
// LLVM:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[ELT2]], i32 noundef 3)
// LLVM-NEXT: to label %{{.*}} unwind label %[[LPAD]]
// LLVM: [[LPAD]]:
// LLVM-NEXT: landingpad { ptr, i32 }
// LLVM-NEXT: cleanup
// LLVM:   call void @_ZN6ThrowsD1Ev(
// LLVM:   call void @_ZdaPv(ptr noundef %[[ALLOC]]
// LLVM:   resume { ptr, i32 }

// OGCG: define{{.*}} void @_Z19cleanup_const_exactv(
// OGCG-SAME: personality ptr @__gxx_personality_v0
// OGCG:   %[[INIT_END:.*]] = alloca ptr
// OGCG:   %[[ALLOC:.*]] = call{{.*}}ptr @_Znam(i64 noundef 11)
// OGCG:   store i64 3, ptr %[[ALLOC]]
// OGCG:   %[[FIRST:.*]] = getelementptr inbounds i8, ptr %[[ALLOC]], i64 8
// OGCG:   store ptr %[[FIRST]], ptr %[[INIT_END]]
// OGCG:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[FIRST]], i32 noundef 1)
// OGCG-NEXT: to label %[[CONT0:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[CONT0]]:
// OGCG:   %[[ELT1:.*]] = getelementptr inbounds %struct.Throws, ptr %[[FIRST]], i64 1
// OGCG:   store ptr %[[ELT1]], ptr %[[INIT_END]]
// OGCG:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[ELT1]], i32 noundef 2)
// OGCG-NEXT: to label %[[CONT1:.*]] unwind label %[[LPAD]]
// OGCG: [[CONT1]]:
// OGCG:   %[[ELT2:.*]] = getelementptr inbounds %struct.Throws, ptr %[[ELT1]], i64 1
// OGCG:   store ptr %[[ELT2]], ptr %[[INIT_END]]
// OGCG:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[ELT2]], i32 noundef 3)
// OGCG-NEXT: to label %[[CONT2:.*]] unwind label %[[LPAD]]
// OGCG: [[CONT2]]:
// OGCG:   %{{.*}} = getelementptr inbounds %struct.Throws, ptr %[[ELT2]], i64 1
// OGCG:   ret void
//
// Single landing pad that cleans up however many elements were constructed
// using %array.init.end as the upper bound, then calls operator delete[].
// OGCG: [[LPAD]]:
// OGCG-NEXT: landingpad { ptr, i32 }
// OGCG-NEXT: cleanup
// OGCG:   %[[END:.*]] = load ptr, ptr %[[INIT_END]]
// OGCG:   %[[EMPTY:.*]] = icmp eq ptr %[[FIRST]], %[[END]]
// OGCG:   br i1 %[[EMPTY]], label %{{.*}}, label %[[DTOR_BODY:.*]]
// OGCG: [[DTOR_BODY]]:
// OGCG:   %[[PAST:.*]] = phi ptr [ %[[END]], %[[LPAD]] ]
// OGCG-SAME: , [ %[[ELT:.*]], %[[DTOR_BODY]] ]
// OGCG:   %[[ELT]] = getelementptr inbounds %struct.Throws, ptr %[[PAST]], i64 -1
// OGCG:   call void @_ZN6ThrowsD1Ev(ptr {{[^,]*}} %[[ELT]])
// OGCG:   call void @_ZdaPv(ptr noundef %[[ALLOC]])
// OGCG:   resume { ptr, i32 }

// Constant-bound array new where the bound (5) is larger than the number of
// initialisers (3). The first three elements are initialised from the
// init-list and the remaining two are default-constructed in a fixed-trip
// constructor loop.
//
// Classic codegen ends up with two landing pads here (one for the init list
// and one for the loop) and a FIXME noting they could be merged. In CIR we
// thread arrayinit.endOfInit through the constructor loop body, so the
// existing irregular partial-array cleanup covers loop-constructed elements
// as well and the cir.array.ctor lowering does not emit a separate
// partial-destruction EH scope. The result is a single landing pad in the
// LLVM output and no inner cir.cleanup.scope around the loop in CIR.

void cleanup_const_partial() { new Throws[5]{1, 2, 3}; }

// CIR-LABEL: cir.func{{.*}} @_Z21cleanup_const_partialv(
// CIR:   %[[INIT_END:.*]] = cir.alloca {{.*}}, !cir.ptr<!cir.ptr<!rec_Throws>>, ["arrayinit.endOfInit"]
// CIR:   %[[ELT0:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!void> -> !cir.ptr<!rec_Throws>
// CIR:   cir.cleanup.scope {
// CIR:     cir.store align(8) %[[ELT0]], %[[INIT_END]]
// CIR:     cir.call @_ZN6ThrowsC1Ei(%[[ELT0]], %{{.*}})
// CIR:     %[[ELT1:.*]] = cir.ptr_stride %[[ELT0]], %{{.*}}
// CIR:     cir.store align(8) %[[ELT1]], %[[INIT_END]]
// CIR:     cir.call @_ZN6ThrowsC1Ei(%[[ELT1]], %{{.*}})
// CIR:     %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %{{.*}}
// CIR:     cir.store align(8) %[[ELT2]], %[[INIT_END]]
// CIR:     cir.call @_ZN6ThrowsC1Ei(%[[ELT2]], %{{.*}})
// CIR:     %[[REST_BEGIN:.*]] = cir.ptr_stride %[[ELT2]], %{{.*}}
//
// Store the start of the constructor-loop range to endOfInit too, so the
// irregular cleanup that's still active during the loop knows where the
// init-list ended.
// CIR:     cir.store align(8) %[[REST_BEGIN]], %[[INIT_END]]
//
// The constructor loop is emitted directly inside the irregular cleanup
// scope -- there is no nested cir.cleanup.scope here. The loop body stores
// the current element pointer to arrayinit.endOfInit before each ctor call
// so the existing irregular cleanup destroys [BeginPtr, *endOfInit) on
// throw and covers loop-constructed elements as well.
// CIR-NOT:       cir.cleanup.scope
// CIR:     cir.do {
// CIR:       %[[CUR:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!rec_Throws>>, !cir.ptr<!rec_Throws>
// CIR:       cir.store align(8) %[[CUR]], %[[INIT_END]]
// CIR:       cir.call @_ZN6ThrowsC1Ev(%[[CUR]])
// CIR:     } while {
// CIR:       cir.cmp ne %{{.*}}, %{{.*}}
// CIR:     }
// CIR:   } cleanup eh {
// CIR:     %[[END:.*]] = cir.load align(8) %[[INIT_END]]
// CIR:     cir.cmp ne %[[END]], %[[ELT0]]
// CIR:     cir.call @_ZN6ThrowsD1Ev(%{{.*}}) nothrow
// CIR:   }
// CIR:   cir.call @_ZdaPv(

// In the lowered LLVM, all four ctor invokes (three init-list elements plus
// the loop body's default ctor) share a single landing pad. The cleanup
// destroys [BeginPtr, *arrayinit.endOfInit) and then calls operator delete[].
// LLVM-LABEL: define{{.*}} void @_Z21cleanup_const_partialv(
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:   %[[ALLOC:.*]] = call{{.*}}ptr @_Znam(i64 noundef 13)
// LLVM:   store i64 5, ptr %[[ALLOC]]
// LLVM:   %[[FIRST:.*]] = getelementptr i8, ptr %[[ALLOC]], i64 8
// LLVM:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[FIRST]], i32 noundef 1)
// LLVM-NEXT: to label %{{.*}} unwind label %[[LPAD:.*]]
// LLVM:   %[[ELT1:.*]] = getelementptr %struct.Throws, ptr %[[FIRST]], i64 1
// LLVM:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[ELT1]], i32 noundef 2)
// LLVM-NEXT: to label %{{.*}} unwind label %[[LPAD]]
// LLVM:   %[[ELT2:.*]] = getelementptr %struct.Throws, ptr %[[ELT1]], i64 1
// LLVM:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[ELT2]], i32 noundef 3)
// LLVM-NEXT: to label %{{.*}} unwind label %[[LPAD]]
//
// Loop body invokes the default ctor and unwinds to the *same* landing pad.
// LLVM:   invoke void @_ZN6ThrowsC1Ev(
// LLVM-NEXT: to label %{{.*}} unwind label %[[LPAD]]
//
// Single landing pad shared by all four invokes.
// LLVM: [[LPAD]]:
// LLVM-NEXT: landingpad { ptr, i32 }
// LLVM-NEXT: cleanup
// LLVM:   call void @_ZN6ThrowsD1Ev(
// LLVM:   call void @_ZdaPv(ptr noundef %[[ALLOC]]
// LLVM:   resume { ptr, i32 }

// OGCG: define{{.*}} void @_Z21cleanup_const_partialv(
// OGCG-SAME: personality ptr @__gxx_personality_v0
// OGCG:   %[[INIT_END:.*]] = alloca ptr
// OGCG:   %[[ALLOC:.*]] = call{{.*}}ptr @_Znam(i64 noundef 13)
// OGCG:   store i64 5, ptr %[[ALLOC]]
// OGCG:   %[[FIRST:.*]] = getelementptr inbounds i8, ptr %[[ALLOC]], i64 8
// OGCG:   store ptr %[[FIRST]], ptr %[[INIT_END]]
// OGCG:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[FIRST]], i32 noundef 1)
// OGCG-NEXT: to label %[[CONT0:.*]] unwind label %[[LPAD_INIT:.*]]
// OGCG: [[CONT0]]:
// OGCG:   %[[ELT1:.*]] = getelementptr inbounds %struct.Throws, ptr %[[FIRST]], i64 1
// OGCG:   store ptr %[[ELT1]], ptr %[[INIT_END]]
// OGCG:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[ELT1]], i32 noundef 2)
// OGCG-NEXT: to label %[[CONT1:.*]] unwind label %[[LPAD_INIT]]
// OGCG: [[CONT1]]:
// OGCG:   %[[ELT2:.*]] = getelementptr inbounds %struct.Throws, ptr %[[ELT1]], i64 1
// OGCG:   store ptr %[[ELT2]], ptr %[[INIT_END]]
// OGCG:   invoke void @_ZN6ThrowsC1Ei(ptr {{[^,]*}} %[[ELT2]], i32 noundef 3)
// OGCG-NEXT: to label %[[CONT2:.*]] unwind label %[[LPAD_INIT]]
// OGCG: [[CONT2]]:
// OGCG:   %[[REST_BEGIN:.*]] = getelementptr inbounds %struct.Throws, ptr %[[ELT2]], i64 1
// OGCG:   store ptr %[[REST_BEGIN]], ptr %[[INIT_END]]
//
// Default-construct the remaining two elements with a fixed-trip loop. Unlike
// the VLA case, there is no leading isempty check because the trip count is a
// non-zero constant.
// OGCG:   %[[CTOR_END:.*]] = getelementptr inbounds %struct.Throws, ptr %[[REST_BEGIN]], i64 2
// OGCG:   br label %[[CTOR_LOOP:.*]]
// OGCG: [[CTOR_LOOP]]:
// OGCG:   %[[CUR:.*]] = phi ptr [ %[[REST_BEGIN]], %{{.*}} ], [ %[[NEXT:.*]], %{{.*}} ]
// OGCG:   invoke void @_ZN6ThrowsC1Ev(ptr {{[^,]*}} %[[CUR]])
// OGCG-NEXT: to label %{{.*}} unwind label %[[LPAD_LOOP:.*]]
// OGCG:   %[[NEXT]] = getelementptr inbounds %struct.Throws, ptr %[[CUR]], i64 1
// OGCG:   %[[CTOR_EQ:.*]] = icmp eq ptr %[[NEXT]], %[[CTOR_END]]
// OGCG:   br i1 %[[CTOR_EQ]], label %[[CTOR_DONE:.*]], label %[[CTOR_LOOP]]
// OGCG: [[CTOR_DONE]]:
// OGCG:   ret void
//
// Init-list landing pad: cleans up the initialised prefix using
// %array.init.end and then calls operator delete[].
// OGCG: [[LPAD_INIT]]:
// OGCG-NEXT: landingpad { ptr, i32 }
// OGCG-NEXT: cleanup
//
// Loop landing pad: destroys the loop-constructed elements, then falls into
// the init-list cleanup.
// OGCG: [[LPAD_LOOP]]:
// OGCG-NEXT: landingpad { ptr, i32 }
// OGCG-NEXT: cleanup
// OGCG:   %[[LOOP_EMPTY:.*]] = icmp eq ptr %[[REST_BEGIN]], %[[CUR]]
// OGCG:   br i1 %[[LOOP_EMPTY]], label %{{.*}}, label %[[LOOP_DTOR_BODY:.*]]
// OGCG: [[LOOP_DTOR_BODY]]:
// OGCG:   %[[LOOP_PAST:.*]] = phi ptr [ %[[CUR]], %[[LPAD_LOOP]] ]
// OGCG-SAME: , [ %[[LOOP_ELT:.*]], %[[LOOP_DTOR_BODY]] ]
// OGCG:   %[[LOOP_ELT]] = getelementptr inbounds %struct.Throws, ptr %[[LOOP_PAST]], i64 -1
// OGCG:   call void @_ZN6ThrowsD1Ev(ptr {{[^,]*}} %[[LOOP_ELT]])
//
// Init-list cleanup body: pops elements from current down to the allocation
// base using array.init.end, then calls operator delete[] and resumes.
// OGCG:   %[[END:.*]] = load ptr, ptr %[[INIT_END]]
// OGCG:   %[[INIT_EMPTY:.*]] = icmp eq ptr %[[FIRST]], %[[END]]
// OGCG:   br i1 %[[INIT_EMPTY]], label %{{.*}}, label %[[INIT_DTOR_BODY:.*]]
// OGCG: [[INIT_DTOR_BODY]]:
// OGCG:   %[[INIT_PAST:.*]] = phi ptr
// OGCG:   %[[INIT_ELT:.*]] = getelementptr inbounds %struct.Throws, ptr %[[INIT_PAST]], i64 -1
// OGCG:   call void @_ZN6ThrowsD1Ev(ptr {{[^,]*}} %[[INIT_ELT]])
// OGCG:   call void @_ZdaPv(ptr noundef %[[ALLOC]])
// OGCG:   resume { ptr, i32 }
