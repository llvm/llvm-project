// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct StructWithDestructor {
  ~StructWithDestructor();
};

void use(StructWithDestructor &a);

void test_goto_within_cleanup(bool cond) {
  StructWithDestructor a;
  if (cond)
    goto end;
  use(a);
end:
  use(a);
}

// CIR-LABEL: cir.func {{.*}} @_Z24test_goto_within_cleanupb
// CIR:         %[[A_ADDR:.*]] = cir.alloca !rec_StructWithDestructor, !cir.ptr<!rec_StructWithDestructor>, ["a"]
// CIR:         cir.cleanup.scope {
// CIR:           cir.scope {
// CIR:             cir.if {{.*}} {
// CIR:               cir.goto "end"
// CIR:             }
// CIR:           }
// CIR:           cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR:           cir.br ^[[BB_LABEL:bb[0-9]+]]
// CIR:         ^[[BB_LABEL]]:
// CIR:           cir.label "end"
// CIR:           cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR:           cir.yield
// CIR:         } cleanup normal {
// CIR:           cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]])
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define dso_local void @_Z24test_goto_within_cleanupb(i1 noundef %{{.*}})
// LLVM:         %[[COND_ADDR:.*]] = alloca i8
// LLVM:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         %[[COND_BYTE:.*]] = load i8, ptr %[[COND_ADDR]]
// LLVM:         %[[COND_BIT:.*]] = trunc i8 %[[COND_BYTE]] to i1
// LLVM:         br i1 %[[COND_BIT]], label %[[GOTO_BB:.*]], label %[[FALLTHROUGH:.*]]
// LLVM:       [[GOTO_BB]]:
// LLVM:         br label %[[LABEL_BB:[0-9]+]]
// LLVM:       [[FALLTHROUGH]]:
// LLVM:         br label %[[USE_BB:.*]]
// LLVM:       [[USE_BB]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM:         br label %[[LABEL_BB]]
// LLVM:       [[LABEL_BB]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z24test_goto_within_cleanupb(i1 noundef zeroext %cond)
// OGCG:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// OGCG:         br i1 %{{.*}}, label %if.then, label %if.end
// OGCG:       if.then:
// OGCG:         br label %end
// OGCG:       if.end:
// OGCG:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// OGCG:         br label %end
// OGCG:       end:
// OGCG:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// OGCG:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// OGCG-NOT:     call void @_ZN20StructWithDestructorD1Ev
// OGCG:         ret void

void test_goto_jump_into_nested_op(bool cond1, bool cond2) {
  StructWithDestructor a;
  if (cond1)
    goto skip;
  use(a);
  if (cond2) {
skip:
    use(a);
  }
}

// CIR-LABEL: cir.func {{.*}} @_Z29test_goto_jump_into_nested_opbb
// CIR:         %[[A_ADDR:.*]] = cir.alloca !rec_StructWithDestructor, !cir.ptr<!rec_StructWithDestructor>, ["a"]
// CIR:         cir.cleanup.scope {
// CIR:           cir.scope {
// CIR:             cir.if {{.*}} {
// CIR:               cir.goto "skip"
// CIR:             }
// CIR:           }
// CIR:           cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR:           cir.scope {
// CIR:             cir.if {{.*}} {
// CIR:               cir.br ^[[BB_LABEL:bb[0-9]+]]
// CIR:             ^[[BB_LABEL]]:
// CIR:               cir.label "skip"
// CIR:               cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR:               cir.yield
// CIR:             }
// CIR:           }
// CIR:           cir.yield
// CIR:         } cleanup normal {
// CIR:           cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]])
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define dso_local void @_Z29test_goto_jump_into_nested_opbb(i1 noundef %{{.*}}, i1 noundef %{{.*}})
// LLVM:         %[[COND1_ADDR:.*]] = alloca i8
// LLVM:         %[[COND2_ADDR:.*]] = alloca i8
// LLVM:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         %[[COND1_BYTE:.*]] = load i8, ptr %[[COND1_ADDR]]
// LLVM:         %[[COND1_BIT:.*]] = trunc i8 %[[COND1_BYTE]] to i1
// LLVM:         br i1 %[[COND1_BIT]], label %[[GOTO_BB:.*]], label %[[FALLTHROUGH:.*]]
// LLVM:       [[GOTO_BB]]:
// LLVM:         br label %[[SKIP_BB:[0-9]+]]
// LLVM:       [[FALLTHROUGH]]:
// LLVM:         br label %[[USE_BB:.*]]
// LLVM:       [[USE_BB]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM:         br label %[[COND2_BB:.*]]
// LLVM:       [[COND2_BB]]:
// LLVM:         %[[COND2_BYTE:.*]] = load i8, ptr %[[COND2_ADDR]]
// LLVM:         %[[COND2_BIT:.*]] = trunc i8 %[[COND2_BYTE]] to i1
// LLVM:         br i1 %[[COND2_BIT]], label %[[GOTO_BB2:.*]], label %[[MERGE:.*]]
// LLVM:       [[GOTO_BB2]]:
// LLVM:         br label %[[SKIP_BB]]
// LLVM:       [[SKIP_BB]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM:         br label %[[MERGE]]
// LLVM:       [[MERGE]]:
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z29test_goto_jump_into_nested_opbb(i1 noundef zeroext %cond1, i1 noundef zeroext %cond2)
// OGCG:         %[[COND1_ADDR:.*]] = alloca i8
// OGCG:         %[[COND2_ADDR:.*]] = alloca i8
// OGCG:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// OGCG:         %[[COND1_BYTE:.*]] = load i8, ptr %[[COND1_ADDR]]
// OGCG:         %[[COND1_BIT:.*]] = icmp ne i8 %[[COND1_BYTE]], 0
// OGCG:         br i1 %[[COND1_BIT]], label %if.then, label %if.end
// OGCG:       if.then:
// OGCG:         br label %skip
// OGCG:       if.end:
// OGCG:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// OGCG:         %[[COND2_BYTE:.*]] = load i8, ptr %[[COND2_ADDR]]
// OGCG:         %[[COND2_BIT:.*]] = icmp ne i8 %[[COND2_BYTE]], 0
// OGCG:         br i1 %[[COND2_BIT]], label %if.then{{[0-9]+}}, label %[[ENDIF:if\.end[0-9]+]]
// OGCG:       if.then{{[0-9]+}}:
// OGCG:         br label %skip
// OGCG:       skip:
// OGCG:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// OGCG:         br label %[[ENDIF]]
// OGCG:       [[ENDIF]]:
// OGCG:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// OGCG-NOT:     call void @_ZN20StructWithDestructorD1Ev
// OGCG:         ret void
