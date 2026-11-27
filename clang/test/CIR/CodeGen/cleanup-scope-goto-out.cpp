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

void test_goto_only_exit(bool cond) {
  if (cond) {
    StructWithDestructor a;
    use(a);
    goto end;
  }
end:;
}

// CIR-LABEL: cir.func {{.*}} @_Z19test_goto_only_exitb
// CIR:         %[[SLOT:.*]] = cir.alloca "__goto_dest_slot"
// CIR:         cir.scope {
// CIR:           cir.if {{.*}} {
// CIR:             %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR:             cir.cleanup.scope {
// CIR:               cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR:               %[[ID:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:          cir.store %[[ID]], %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:          cir.yield
// CIR:             } cleanup normal {
// CIR:               cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]])
// CIR:               cir.yield
// CIR:             }
// CIR:             cir.switch.flat %{{.+}} : !s32i, ^[[IF_EXIT:bb[0-9]+]] [
// CIR-NEXT:          1: ^[[LEAVE_IF:bb[0-9]+]]
// CIR:           ^[[LEAVE_IF]]:
// CIR-NEXT:        cir.br ^[[IF_EXIT]]
// CIR:           ^[[IF_EXIT]]:
// CIR-NEXT:        cir.yield
// CIR:           }
// CIR:           cir.switch.flat %{{.+}} : !s32i, ^[[SCOPE_EXIT:bb[0-9]+]] [
// CIR-NEXT:        1: ^[[LEAVE_SCOPE:bb[0-9]+]]
// CIR:         ^[[LEAVE_SCOPE]]:
// CIR-NEXT:      cir.br ^[[SCOPE_EXIT]]
// CIR:         ^[[SCOPE_EXIT]]:
// CIR-NEXT:      cir.yield
// CIR:         }
// CIR:         cir.switch.flat %{{.+}} : !s32i, ^[[CONT:bb[0-9]+]] [
// CIR-NEXT:      1: ^[[GOTO_DEST:bb[0-9]+]]
// CIR:       ^[[GOTO_DEST]]:
// CIR:         cir.store %{{.+}}, %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:    cir.br ^[[END_BB:bb[0-9]+]]
// CIR:       ^[[CONT]]:
// CIR-NEXT:    cir.br ^[[END_BB]]
// CIR:       ^[[END_BB]]:
// CIR-NEXT:    cir.return

// LLVM-LABEL: define dso_local void @_Z19test_goto_only_exitb(i1 noundef %{{[^,)]+}})
// LLVM:         %[[DEST_SLOT:.*]] = alloca i32
// LLVM:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         br i1 %{{.*}}, label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// LLVM:       [[IF_THEN]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM-NEXT:    store i32 0, ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    br label %[[CLEANUP:.*]]
// LLVM:       [[CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         [[D:%.*]] = load i32, ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    switch i32 [[D]], label %[[UNREACHABLE:.*]] [
// LLVM-NEXT:      i32 0, label %[[GOTO_DEST:.*]]
// LLVM:         ]
// LLVM:       [[UNREACHABLE]]:
// LLVM-NEXT:    unreachable
// LLVM:       [[GOTO_DEST]]:
// LLVM-NEXT:    br label %[[END:.*]]
// LLVM:       [[IF_END]]:
// LLVM-NEXT:    br label %[[END]]
// LLVM:       [[END]]:
// LLVM-NEXT:    ret void

// OGCG-LABEL: define dso_local void @_Z19test_goto_only_exitb(i1 noundef %{{.*}})
// OGCG:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// OGCG:         br i1 %{{.*}}, label %if.then, label %if.end
// OGCG:       if.then:
// OGCG:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// OGCG:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// OGCG-NOT:     call void @_ZN20StructWithDestructorD1Ev
// OGCG:         switch i32 %{{.*}}, label %unreachable [
// OGCG:           i32 2, label %end
// OGCG:         ]
// OGCG:       end:
// OGCG:         ret void

void test_goto_among_other_exits(bool cond) {
  {
    StructWithDestructor a;
    if (cond)
      goto end;
    use(a);
  }
end:;
}

// CIR-LABEL: cir.func {{.*}} @_Z27test_goto_among_other_exitsb
// CIR:         %[[SLOT:.*]] = cir.alloca "__goto_dest_slot"
// CIR:         cir.scope {
// CIR:           %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR:           cir.cleanup.scope {
// CIR:             cir.scope {
// CIR:               cir.if {{.*}} {
// CIR:                 %[[ID:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:            cir.store %[[ID]], %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:          }
// CIR:               cir.switch.flat %{{.+}} : !s32i, ^[[SCOPE_EXIT:bb[0-9]+]] [
// CIR-NEXT:            1: ^[[LEAVE_SCOPE:bb[0-9]+]]
// CIR:             ^[[LEAVE_SCOPE]]:
// CIR-NEXT:          cir.br ^[[SCOPE_EXIT]]
// CIR:             ^[[SCOPE_EXIT]]:
// CIR-NEXT:          cir.yield
// CIR:             }
// CIR:             cir.switch.flat %{{.+}} : !s32i, ^[[USE_BB:bb[0-9]+]] [
// CIR-NEXT:          1: ^[[LEAVE_BODY:bb[0-9]+]]
// CIR:           ^[[LEAVE_BODY]]:
// CIR-NEXT:        cir.yield
// CIR:           ^[[USE_BB]]:
// CIR-NEXT:        cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR-NEXT:        cir.yield
// CIR:           } cleanup normal {
// CIR:             cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]])
// CIR:             cir.yield
// CIR:           }
// CIR:           cir.switch.flat %{{.+}} : !s32i, ^[[SCOPE_EXIT:bb[0-9]+]] [
// CIR-NEXT:        1: ^[[LEAVE_SCOPE:bb[0-9]+]]
// CIR:         ^[[LEAVE_SCOPE]]:
// CIR-NEXT:      cir.br ^[[SCOPE_EXIT]]
// CIR:         ^[[SCOPE_EXIT]]:
// CIR-NEXT:      cir.yield
// CIR:         }
// CIR:         cir.switch.flat %{{.+}} : !s32i, ^[[CONT:bb[0-9]+]] [
// CIR-NEXT:      1: ^[[GOTO_DEST:bb[0-9]+]]
// CIR:       ^[[GOTO_DEST]]:
// CIR:         cir.store %{{.+}}, %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:    cir.br ^[[END_BB:bb[0-9]+]]
// CIR:       ^[[CONT]]:
// CIR-NEXT:    cir.br ^[[END_BB]]
// CIR:       ^[[END_BB]]:
// CIR-NEXT:    cir.return

// LLVM-LABEL: define dso_local void @_Z27test_goto_among_other_exitsb(i1 noundef %{{[^,)]+}})
// LLVM:         %[[DEST_SLOT:.*]] = alloca i32
// LLVM:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         br i1 %{{.*}}, label %[[GOTO_PATH:.*]], label %[[NORMAL_PATH:.*]]
// LLVM:       [[GOTO_PATH]]:
// LLVM-NEXT:    store i32 [[GOTO_ID:[0-9]+]], ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    br label %[[CLEANUP:.*]]
// LLVM:       [[NORMAL_PATH]]:
// LLVM-NEXT:    call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM-NEXT:    store i32 [[FALL_ID:[0-9]+]], ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    br label %[[CLEANUP]]
// LLVM:       [[CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         [[D:%.*]] = load i32, ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    switch i32 [[D]], label %[[UNREACHABLE:.*]] [
// LLVM-NEXT:      i32 [[GOTO_ID]], label %[[GOTO_DEST:.*]]
// LLVM-NEXT:      i32 [[FALL_ID]], label %[[FALL_DEST:.*]]
// LLVM-NEXT:    ]
// LLVM:       [[UNREACHABLE]]:
// LLVM-NEXT:    unreachable
// LLVM:       [[GOTO_DEST]]:
// LLVM-NEXT:    br label %[[END:.*]]
// LLVM:       [[FALL_DEST]]:
// LLVM-NEXT:    br label %[[END]]
// LLVM:       [[END]]:
// LLVM-NEXT:    ret void

// OGCG-LABEL: define dso_local void @_Z27test_goto_among_other_exitsb(i1 noundef %{{.*}})
// OGCG:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// OGCG:         br i1 %{{.*}}, label %if.then, label %if.end
// OGCG:       if.then:
// OGCG:         store i32 2, ptr %{{.*}}
// OGCG:         br label %cleanup
// OGCG:       if.end:
// OGCG:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// OGCG:         store i32 0, ptr %{{.*}}
// OGCG:         br label %cleanup
// OGCG:       cleanup:
// OGCG:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// OGCG-NOT:     call void @_ZN20StructWithDestructorD1Ev
// OGCG:         switch i32 %{{.*}}, label %unreachable [
// OGCG:           i32 0, label %cleanup.cont
// OGCG:           i32 2, label %end
// OGCG:         ]
// OGCG:       end:
// OGCG:         ret void

void test_goto_inner_to_outer(bool cond) {
  StructWithDestructor outer;
  {
    StructWithDestructor inner;
    if (cond)
      goto skip;
    use(inner);
  }
skip:
  use(outer);
}

// CIR-LABEL: cir.func {{.*}} @_Z24test_goto_inner_to_outerb
// CIR:         %[[SLOT:.*]] = cir.alloca "__goto_dest_slot"
// CIR:         %[[OUTER_ADDR:.*]] = cir.alloca "outer" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR:         cir.cleanup.scope {
// CIR:           cir.scope {
// CIR:             %[[INNER_ADDR:.*]] = cir.alloca "inner" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR:             cir.cleanup.scope {
// CIR:               cir.scope {
// CIR:                 cir.if {{.*}} {
// CIR:                   %[[ID:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:              cir.store %[[ID]], %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:            }
// CIR:                 cir.switch.flat %{{.+}} : !s32i, ^[[SCOPE_EXIT:bb[0-9]+]] [
// CIR-NEXT:              1: ^[[LEAVE_SCOPE:bb[0-9]+]]
// CIR:               ^[[LEAVE_SCOPE]]:
// CIR-NEXT:            cir.br ^[[SCOPE_EXIT]]
// CIR:               ^[[SCOPE_EXIT]]:
// CIR-NEXT:            cir.yield
// CIR:               }
// CIR:               cir.switch.flat %{{.+}} : !s32i, ^[[USE_BB:bb[0-9]+]] [
// CIR-NEXT:            1: ^[[LEAVE_BODY:bb[0-9]+]]
// CIR:             ^[[LEAVE_BODY]]:
// CIR-NEXT:          cir.yield
// CIR:             ^[[USE_BB]]:
// CIR-NEXT:          cir.call @_Z3useR20StructWithDestructor(%[[INNER_ADDR]])
// CIR-NEXT:          cir.yield
// CIR:             } cleanup normal {
// CIR:               cir.call @_ZN20StructWithDestructorD1Ev(%[[INNER_ADDR]])
// CIR:               cir.yield
// CIR:             }
// CIR:             cir.switch.flat %{{.+}} : !s32i, ^[[SCOPE_EXIT:bb[0-9]+]] [
// CIR-NEXT:          1: ^[[LEAVE_SCOPE:bb[0-9]+]]
// CIR:           ^[[LEAVE_SCOPE]]:
// CIR-NEXT:        cir.br ^[[SCOPE_EXIT]]
// CIR:           ^[[SCOPE_EXIT]]:
// CIR-NEXT:        cir.yield
// CIR:           }
// CIR:           cir.switch.flat %{{.+}} : !s32i, ^[[CONT:bb[0-9]+]] [
// CIR-NEXT:        1: ^[[GOTO_DEST:bb[0-9]+]]
// CIR:         ^[[GOTO_DEST]]:
// CIR:           cir.store %{{.+}}, %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:      cir.br ^[[SKIP_BB:bb[0-9]+]]
// CIR:         ^[[CONT]]:
// CIR-NEXT:      cir.br ^[[SKIP_BB]]
// CIR:         ^[[SKIP_BB]]:
// CIR-NEXT:      cir.call @_Z3useR20StructWithDestructor(%[[OUTER_ADDR]])
// CIR-NEXT:      cir.yield
// CIR:         } cleanup normal {
// CIR:           cir.call @_ZN20StructWithDestructorD1Ev(%[[OUTER_ADDR]])
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define dso_local void @_Z24test_goto_inner_to_outerb(i1 noundef %{{[^,)]+}})
// LLVM:         %[[DEST_SLOT:.*]] = alloca i32
// LLVM:         br i1 %{{.*}}, label %[[GOTO_PATH:.*]], label %[[NORMAL_PATH:.*]]
// LLVM:       [[GOTO_PATH]]:
// LLVM-NEXT:    store i32 [[GOTO_ID:[0-9]+]], ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    br label %[[INNER_CLEANUP:.*]]
// LLVM:       [[NORMAL_PATH]]:
// LLVM-NEXT:    call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[INNER_ADDR:.*]])
// LLVM-NEXT:    store i32 [[FALL_ID:[0-9]+]], ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    br label %[[INNER_CLEANUP]]
// LLVM:       [[INNER_CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[INNER_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         [[D:%.*]] = load i32, ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    switch i32 [[D]], label %[[UNREACHABLE:.*]] [
// LLVM-NEXT:      i32 [[GOTO_ID]], label %[[GOTO_DEST:.*]]
// LLVM-NEXT:      i32 [[FALL_ID]], label %[[FALL_DEST:.*]]
// LLVM-NEXT:    ]
// LLVM:       [[UNREACHABLE]]:
// LLVM-NEXT:    unreachable
// LLVM:       [[GOTO_DEST]]:
// LLVM-NEXT:    br label %[[USE_OUTER:.*]]
// LLVM:       [[FALL_DEST]]:
// LLVM-NEXT:    br label %[[USE_OUTER]]
// LLVM:       [[USE_OUTER]]:
// LLVM-NEXT:    call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[OUTER_ADDR:.*]])
// LLVM:         br label %[[OUTER_DTOR_BB:.*]]
// LLVM:       [[OUTER_DTOR_BB]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[OUTER_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z24test_goto_inner_to_outerb(i1 noundef %{{.*}})
// OGCG:         %[[OUTER_ADDR:.*]] = alloca %struct.StructWithDestructor
// OGCG:         %[[INNER_ADDR:.*]] = alloca %struct.StructWithDestructor
// OGCG:         br i1 %{{.*}}, label %if.then, label %if.end
// OGCG:       if.then:
// OGCG:         store i32 2, ptr %{{.*}}
// OGCG:         br label %cleanup
// OGCG:       if.end:
// OGCG:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[INNER_ADDR]])
// OGCG:         store i32 0, ptr %{{.*}}
// OGCG:         br label %cleanup
// OGCG:       cleanup:
// OGCG:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[INNER_ADDR]])
// OGCG:         switch i32 %{{.*}}, label %unreachable [
// OGCG:           i32 0, label %cleanup.cont
// OGCG:           i32 2, label %skip
// OGCG:         ]
// OGCG:       skip:
// OGCG:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[OUTER_ADDR]])
// OGCG:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[OUTER_ADDR]])
// OGCG-NOT:     call void @_ZN20StructWithDestructorD1Ev
// OGCG:         ret void

void test_goto_out_of_two_nested(bool cond) {
  {
    StructWithDestructor outer;
    {
      StructWithDestructor inner;
      if (cond)
        goto end;
      use(inner);
    }
    use(outer);
  }
end:;
}

// CIR-LABEL: cir.func {{.*}} @_Z27test_goto_out_of_two_nestedb
// CIR:         %[[SLOT:.*]] = cir.alloca "__goto_dest_slot"
// CIR:         cir.scope {
// CIR:           %[[OUTER_ADDR:.*]] = cir.alloca "outer" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR:           cir.cleanup.scope {
// CIR:             cir.scope {
// CIR:               %[[INNER_ADDR:.*]] = cir.alloca "inner" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR:               cir.cleanup.scope {
// CIR:                 cir.scope {
// CIR:                   cir.if {{.*}} {
// CIR:                     %[[ID:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:                cir.store %[[ID]], %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:              }
// CIR:                   cir.switch.flat %{{.+}} : !s32i, ^[[SCOPE_EXIT:bb[0-9]+]] [
// CIR-NEXT:                1: ^[[LEAVE_SCOPE:bb[0-9]+]]
// CIR:                 ^[[LEAVE_SCOPE]]:
// CIR-NEXT:              cir.br ^[[SCOPE_EXIT]]
// CIR:                 ^[[SCOPE_EXIT]]:
// CIR-NEXT:              cir.yield
// CIR:                 }
// CIR:                 cir.switch.flat %{{.+}} : !s32i, ^[[USE_BB:bb[0-9]+]] [
// CIR-NEXT:              1: ^[[LEAVE_BODY:bb[0-9]+]]
// CIR:               ^[[LEAVE_BODY]]:
// CIR-NEXT:            cir.yield
// CIR:               ^[[USE_BB]]:
// CIR-NEXT:            cir.call @_Z3useR20StructWithDestructor(%[[INNER_ADDR]])
// CIR-NEXT:            cir.yield
// CIR:               } cleanup normal {
// CIR:                 cir.call @_ZN20StructWithDestructorD1Ev(%[[INNER_ADDR]])
// CIR:                 cir.yield
// CIR:               }
// CIR:               cir.switch.flat %{{.+}} : !s32i, ^[[SCOPE_EXIT:bb[0-9]+]] [
// CIR-NEXT:            1: ^[[LEAVE_SCOPE:bb[0-9]+]]
// CIR:             ^[[LEAVE_SCOPE]]:
// CIR-NEXT:          cir.br ^[[SCOPE_EXIT]]
// CIR:             ^[[SCOPE_EXIT]]:
// CIR-NEXT:          cir.yield
// CIR:             }
// CIR:             cir.switch.flat %{{.+}} : !s32i, ^[[USE_BB:bb[0-9]+]] [
// CIR-NEXT:          1: ^[[LEAVE_BODY:bb[0-9]+]]
// CIR:           ^[[LEAVE_BODY]]:
// CIR-NEXT:        cir.yield
// CIR:           ^[[USE_BB]]:
// CIR-NEXT:        cir.call @_Z3useR20StructWithDestructor(%[[OUTER_ADDR]])
// CIR-NEXT:        cir.yield
// CIR:           } cleanup normal {
// CIR:             cir.call @_ZN20StructWithDestructorD1Ev(%[[OUTER_ADDR]])
// CIR:             cir.yield
// CIR:           }
// CIR:           cir.switch.flat %{{.+}} : !s32i, ^[[SCOPE_EXIT:bb[0-9]+]] [
// CIR-NEXT:        1: ^[[LEAVE_SCOPE:bb[0-9]+]]
// CIR:         ^[[LEAVE_SCOPE]]:
// CIR-NEXT:      cir.br ^[[SCOPE_EXIT]]
// CIR:         ^[[SCOPE_EXIT]]:
// CIR-NEXT:      cir.yield
// CIR:         }
// CIR:         cir.switch.flat %{{.+}} : !s32i, ^[[CONT:bb[0-9]+]] [
// CIR-NEXT:      1: ^[[GOTO_DEST:bb[0-9]+]]
// CIR:       ^[[GOTO_DEST]]:
// CIR:         cir.store %{{.+}}, %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:    cir.br ^[[END_BB:bb[0-9]+]]
// CIR:       ^[[CONT]]:
// CIR-NEXT:    cir.br ^[[END_BB]]
// CIR:       ^[[END_BB]]:
// CIR-NEXT:    cir.return

// LLVM-LABEL: define dso_local void @_Z27test_goto_out_of_two_nestedb(i1 noundef %{{[^,)]+}})
// LLVM:         %[[DEST_SLOT:.*]] = alloca i32
// LLVM:         %[[OUTER_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         %[[INNER_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         br i1 %{{.*}}, label %[[GOTO_PATH:.*]], label %[[NORMAL_PATH:.*]]
// LLVM:       [[GOTO_PATH]]:
// LLVM-NEXT:    store i32 [[INNER_GOTO_ID:[0-9]+]], ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    br label %[[INNER_CLEANUP:.*]]
// LLVM:       [[NORMAL_PATH]]:
// LLVM-NEXT:    call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[INNER_ADDR]])
// LLVM-NEXT:    store i32 [[INNER_FALL_ID:[0-9]+]], ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    br label %[[INNER_CLEANUP]]
// LLVM:       [[INNER_CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[INNER_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         [[D:%.*]] = load i32, ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    switch i32 [[D]], label %[[UNREACHABLE:.*]] [
// LLVM-NEXT:      i32 [[INNER_GOTO_ID]], label %[[INNER_GOTO:.*]]
// LLVM-NEXT:      i32 [[INNER_FALL_ID]], label %[[USE_OUTER:.*]]
// LLVM-NEXT:    ]
// LLVM:       [[UNREACHABLE]]:
// LLVM-NEXT:    unreachable
// LLVM:       [[INNER_GOTO]]:
// LLVM-NEXT:    store i32 [[OUTER_GOTO_ID:[0-9]+]], ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    br label %[[OUTER_CLEANUP:.*]]
// LLVM:       [[USE_OUTER]]:
// LLVM-NEXT:    call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[OUTER_ADDR]])
// LLVM-NEXT:    store i32 [[OUTER_FALL_ID:[0-9]+]], ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    br label %[[OUTER_CLEANUP]]
// LLVM:       [[OUTER_CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[OUTER_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         [[D:%.*]] = load i32, ptr %[[DEST_SLOT]], align 4
// LLVM-NEXT:    switch i32 [[D]], label %[[UNREACHABLE:.*]] [
// LLVM-NEXT:      i32 [[OUTER_GOTO_ID]], label %[[GOTO_DEST:.*]]
// LLVM-NEXT:      i32 [[OUTER_FALL_ID]], label %[[FALL_DEST:.*]]
// LLVM-NEXT:    ]
// LLVM:       [[UNREACHABLE]]:
// LLVM-NEXT:    unreachable
// LLVM:       [[GOTO_DEST]]:
// LLVM-NEXT:    br label %[[END:.*]]
// LLVM:       [[FALL_DEST]]:
// LLVM-NEXT:    br label %[[END]]
// LLVM:       [[END]]:
// LLVM-NEXT:    ret void

// OGCG-LABEL: define dso_local void @_Z27test_goto_out_of_two_nestedb(i1 noundef %{{.*}})
// OGCG:         %[[OUTER_ADDR:.*]] = alloca %struct.StructWithDestructor
// OGCG:         %[[INNER_ADDR:.*]] = alloca %struct.StructWithDestructor
// OGCG:         br i1 %{{.*}}, label %if.then, label %if.end
// OGCG:       if.then:
// OGCG:         store i32 2, ptr %{{.*}}
// OGCG:         br label %cleanup
// OGCG:       if.end:
// OGCG:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[INNER_ADDR]])
// OGCG:         store i32 0, ptr %{{.*}}
// OGCG:         br label %cleanup
// OGCG:       cleanup:
// OGCG:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[INNER_ADDR]])
// OGCG:         switch i32 %{{.*}}, label %[[OUTER_CLEANUP:.*]] [
// OGCG:           i32 0, label %cleanup.cont
// OGCG:         ]
// OGCG:       cleanup.cont:
// OGCG:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[OUTER_ADDR]])
// OGCG:         store i32 0, ptr %{{.*}}
// OGCG:         br label %[[OUTER_CLEANUP]]
// OGCG:       [[OUTER_CLEANUP]]:
// OGCG:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[OUTER_ADDR]])
// OGCG-NOT:     call void @_ZN20StructWithDestructorD1Ev
// OGCG:         switch i32 %{{.*}}, label %unreachable [
// OGCG:           i32 0, label %cleanup.cont{{[0-9]*}}
// OGCG:           i32 2, label %end
// OGCG:         ]
// OGCG:       end:
// OGCG:         ret void
