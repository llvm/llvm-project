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
// CIR:         %[[SLOT:.*]] = cir.alloca "__goto_dest_slot"
// CIR:         %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR:         cir.cleanup.scope {
// CIR:           cir.scope {
// CIR:             cir.if {{.*}} {
// CIR:               %[[ID:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:          cir.store %[[ID]], %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:        }
// CIR:             cir.switch.flat %{{.+}} : !s32i, ^[[SCOPE_EXIT:bb[0-9]+]] [
// CIR-NEXT:          1: ^[[LEAVE_SCOPE:bb[0-9]+]]
// CIR:           ^[[LEAVE_SCOPE]]:
// CIR-NEXT:        cir.br ^[[SCOPE_EXIT]]
// CIR:           ^[[SCOPE_EXIT]]:
// CIR-NEXT:        cir.yield
// CIR:           }
// CIR:           cir.switch.flat %{{.+}} : !s32i, ^[[USE_BB:bb[0-9]+]] [
// CIR-NEXT:        1: ^[[GOTO_DEST:bb[0-9]+]]
// CIR:         ^[[GOTO_DEST]]:
// CIR:           cir.store %{{.+}}, %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:      cir.br ^[[BB_LABEL:bb[0-9]+]]
// CIR:         ^[[USE_BB]]:
// CIR-NEXT:      cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR-NEXT:      cir.br ^[[BB_LABEL]]
// CIR:         ^[[BB_LABEL]]:
// CIR-NEXT:      cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR:           cir.yield
// CIR:         } cleanup normal {
// CIR:           cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]])
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define dso_local void @_Z24test_goto_within_cleanupb(i1 noundef %{{[^,)]+}})
// LLVM:         %[[COND_ADDR:.*]] = alloca i8
// LLVM:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         %[[COND_BYTE:.*]] = load i8, ptr %[[COND_ADDR]]
// LLVM:         %[[COND_BIT:.*]] = trunc i8 %[[COND_BYTE]] to i1
// LLVM:         br i1 %[[COND_BIT]], label %[[GOTO_BB:.*]], label %[[FALLTHROUGH:.*]]
// LLVM:       [[GOTO_BB]]:
// LLVM-NEXT:    br label %[[LABEL_BB:[0-9]+]]
// LLVM:       [[FALLTHROUGH]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM:         br label %[[LABEL_BB]]
// LLVM:       [[LABEL_BB]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z24test_goto_within_cleanupb(i1 noundef %cond)
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
// CIR:         %[[SLOT:.*]] = cir.alloca "__goto_dest_slot"
// CIR:         %[[COND2_ADDR:.*]] = cir.alloca "cond2"
// CIR:         %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR:         cir.cleanup.scope {
// CIR:           cir.scope {
// CIR:             cir.if {{.*}} {
// CIR:               %[[ID:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:          cir.store %[[ID]], %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:        }
// CIR:           }
// CIR:           cir.switch.flat %{{.+}} : !s32i, ^[[USE_BB:bb[0-9]+]] [
// CIR-NEXT:        1: ^[[GOTO_SKIP:bb[0-9]+]]
// CIR:         ^[[GOTO_SKIP]]:
// CIR-NEXT:      cir.br ^[[PRE_SCOPE:bb[0-9]+]]
// CIR:         ^[[USE_BB]]:
// CIR-NEXT:      cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR:           cir.br ^[[PRE_SCOPE]]
// CIR:         ^[[PRE_SCOPE]]:
// CIR-NEXT:      cir.scope {
// CIR:             %[[DEST:.*]] = cir.load %[[SLOT]]
// CIR-NEXT:        cir.switch.flat %[[DEST]] : !s32i, ^[[NORMAL:bb[0-9]+]] [
// CIR-NEXT:          1: ^[[ENTER_IF:bb[0-9]+]]
// CIR:           ^[[ENTER_IF]]:
// CIR-NEXT:        %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:        cir.br ^[[PRE_IF:bb[0-9]+]](%[[TRUE]] : !cir.bool)
// CIR:           ^[[NORMAL]]:
// CIR-NEXT:        %[[COND2:.*]] = cir.load {{.*}} %[[COND2_ADDR]]
// CIR:             cir.br ^[[PRE_IF]](%[[COND2]] : !cir.bool)
// CIR:           ^[[PRE_IF]](%[[IF_COND:.*]]: !cir.bool
// CIR-NEXT:        cir.if %[[IF_COND]] {
// CIR:               %[[DEST2:.*]] = cir.load %[[SLOT]]
// CIR-NEXT:          cir.switch.flat %[[DEST2]] : !s32i, ^{{bb[0-9]+}} [
// CIR-NEXT:            1: ^[[TO_SKIP:bb[0-9]+]]
// CIR:             ^[[TO_SKIP]]:
// CIR:               cir.br ^[[BB_LABEL:bb[0-9]+]]
// CIR:             ^[[BB_LABEL]]:
// CIR-NEXT:          cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR:               cir.yield
// CIR:             }
// CIR:           }
// CIR:           cir.yield
// CIR:         } cleanup normal {
// CIR:           cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]])
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define dso_local void @_Z29test_goto_jump_into_nested_opbb(i1 noundef %{{[^,)]+}}, i1 noundef %{{[^,)]+}})
// LLVM-NOT:     alloca i32
// LLVM:         %[[COND1_ADDR:.*]] = alloca i8
// LLVM:         %[[COND2_ADDR:.*]] = alloca i8
// LLVM:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         %[[COND1_BYTE:.*]] = load i8, ptr %[[COND1_ADDR]]
// LLVM:         %[[COND1_BIT:.*]] = trunc i8 %[[COND1_BYTE]] to i1
// LLVM:         br i1 %[[COND1_BIT]], label %[[GOTO_BB:.*]], label %[[FALLTHROUGH:.*]]
// LLVM:       [[FALLTHROUGH]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM:         br label %[[COND2_BB:[0-9]+]]
// LLVM:       [[GOTO_BB]]:
// LLVM-NEXT:    br label %[[GOTO_EXIT:[0-9]+]]
// LLVM:       [[COND2_BB]]:
// LLVM:         %[[COND2_BYTE:.*]] = load i8, ptr %[[COND2_ADDR]]
// LLVM:         %[[COND2_BIT:.*]] = trunc i8 %[[COND2_BYTE]] to i1
// LLVM-NEXT:    br i1 %[[COND2_BIT]], label %[[THEN:.*]], label %[[CLEANUP:.*]]
// LLVM:       [[GOTO_EXIT]]:
// LLVM-NEXT:    br label %[[SKIP_BB:[0-9]+]]
// LLVM:       [[THEN]]:
// LLVM-NEXT:    br label %[[SKIP_BB]]
// LLVM:       [[SKIP_BB]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM:         br label %[[CLEANUP]]
// LLVM:       [[CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z29test_goto_jump_into_nested_opbb(i1 noundef %cond1, i1 noundef %cond2)
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
