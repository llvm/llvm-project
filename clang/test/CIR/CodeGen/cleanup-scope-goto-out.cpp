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
// CIR:         cir.scope {
// CIR:           cir.if {{.*}} {
// CIR:             %[[A_ADDR:.*]] = cir.alloca !rec_StructWithDestructor, !cir.ptr<!rec_StructWithDestructor>, ["a"]
// CIR:             cir.cleanup.scope {
// CIR:               cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR:               cir.goto "end"
// CIR:             } cleanup normal {
// CIR:               cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]])
// CIR:               cir.yield
// CIR:             }
// CIR:           }
// CIR:         }
// CIR:         cir.br ^[[END_BB:bb[0-9]+]]
// CIR:       ^[[END_BB]]:
// CIR:         cir.label "end"
// CIR:         cir.return

// LLVM-LABEL: define dso_local void @_Z19test_goto_only_exitb(i1 noundef %{{.*}})
// LLVM:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         br i1 %{{.*}}, label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// LLVM:       [[IF_THEN]]:
// LLVM:         br label %[[BODY:.*]]
// LLVM:       [[BODY]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         switch i32 %{{.*}}, label %[[DEFAULT:.*]] [
// LLVM:           i32 1, label %[[GOTO_DEST:.*]]
// LLVM:         ]
// LLVM:       [[GOTO_DEST]]:
// LLVM:         br label %[[END:.*]]
// LLVM:       [[END]]:
// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z19test_goto_only_exitb(i1 noundef zeroext %{{.*}})
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
// CIR:         cir.scope {
// CIR:           %[[A_ADDR:.*]] = cir.alloca !rec_StructWithDestructor, !cir.ptr<!rec_StructWithDestructor>, ["a"]
// CIR:           cir.cleanup.scope {
// CIR:             cir.scope {
// CIR:               cir.if {{.*}} {
// CIR:                 cir.goto "end"
// CIR:               }
// CIR:             }
// CIR:             cir.call @_Z3useR20StructWithDestructor(%[[A_ADDR]])
// CIR:             cir.yield
// CIR:           } cleanup normal {
// CIR:             cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]])
// CIR:             cir.yield
// CIR:           }
// CIR:         }
// CIR:         cir.br ^[[END_BB:bb[0-9]+]]
// CIR:       ^[[END_BB]]:
// CIR:         cir.label "end"
// CIR:         cir.return

// LLVM-LABEL: define dso_local void @_Z27test_goto_among_other_exitsb(i1 noundef %{{.*}})
// LLVM:         %[[A_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         br i1 %{{.*}}, label %[[GOTO_PATH:.*]], label %[[NORMAL_PATH:.*]]
// LLVM:       [[GOTO_PATH]]:
// LLVM:         store i32 1, ptr %{{.*}}
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[NORMAL_PATH]]:
// LLVM:         br label %[[FALL:.*]]
// LLVM:       [[FALL]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[A_ADDR]])
// LLVM:         store i32 0, ptr %{{.*}}
// LLVM:         br label %[[CLEANUP]]
// LLVM:       [[CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[A_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         switch i32 %{{.*}}, label %[[DEFAULT:.*]] [
// LLVM:           i32 0, label %[[FALL_DEST:.*]]
// LLVM:           i32 1, label %[[GOTO_DEST:.*]]
// LLVM:         ]
// LLVM:       [[FALL_DEST]]:
// LLVM:         br label %[[END:.*]]
// LLVM:       [[END]]:
// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z27test_goto_among_other_exitsb(i1 noundef zeroext %{{.*}})
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
// CIR:         %[[OUTER_ADDR:.*]] = cir.alloca !rec_StructWithDestructor, !cir.ptr<!rec_StructWithDestructor>, ["outer"]
// CIR:         cir.cleanup.scope {
// CIR:           cir.scope {
// CIR:             %[[INNER_ADDR:.*]] = cir.alloca !rec_StructWithDestructor, !cir.ptr<!rec_StructWithDestructor>, ["inner"]
// CIR:             cir.cleanup.scope {
// CIR:               cir.scope {
// CIR:                 cir.if {{.*}} {
// CIR:                   cir.goto "skip"
// CIR:                 }
// CIR:               }
// CIR:               cir.call @_Z3useR20StructWithDestructor(%[[INNER_ADDR]])
// CIR:               cir.yield
// CIR:             } cleanup normal {
// CIR:               cir.call @_ZN20StructWithDestructorD1Ev(%[[INNER_ADDR]])
// CIR:               cir.yield
// CIR:             }
// CIR:           }
// CIR:           cir.br ^[[SKIP_BB:bb[0-9]+]]
// CIR:         ^[[SKIP_BB]]:
// CIR:           cir.label "skip"
// CIR:           cir.call @_Z3useR20StructWithDestructor(%[[OUTER_ADDR]])
// CIR:           cir.yield
// CIR:         } cleanup normal {
// CIR:           cir.call @_ZN20StructWithDestructorD1Ev(%[[OUTER_ADDR]])
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define dso_local void @_Z24test_goto_inner_to_outerb(i1 noundef %{{.*}})
// LLVM:         alloca %struct.StructWithDestructor
// LLVM:         alloca %struct.StructWithDestructor
// LLVM:         br i1 %{{.*}}, label %[[GOTO_PATH:.*]], label %[[NORMAL_PATH:.*]]
// LLVM:       [[GOTO_PATH]]:
// LLVM:         store i32 1, ptr %{{.*}}
// LLVM:         br label %[[INNER_CLEANUP:.*]]
// LLVM:       [[NORMAL_PATH]]:
// LLVM:         br label %[[USE_INNER:.*]]
// LLVM:       [[USE_INNER]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[INNER_ADDR:.*]])
// LLVM:         store i32 0, ptr %{{.*}}
// LLVM:         br label %[[INNER_CLEANUP]]
// LLVM:       [[INNER_CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[INNER_ADDR]])
// LLVM:         switch i32 %{{.*}}, label %[[DEFAULT:.*]] [
// LLVM:           i32 0, label %[[FALL_DEST:.*]]
// LLVM:           i32 1, label %[[GOTO_DEST:.*]]
// LLVM:         ]
// LLVM:       [[FALL_DEST]]:
// LLVM:         br label %{{.*}}
// LLVM:       [[GOTO_DEST]]:
// LLVM:         br label %[[USE_OUTER:.*]]
// LLVM:       [[USE_OUTER]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[OUTER_ADDR:.*]])
// LLVM:         br label %[[OUTER_DTOR_BB:.*]]
// LLVM:       [[OUTER_DTOR_BB]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[OUTER_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z24test_goto_inner_to_outerb(i1 noundef zeroext %{{.*}})
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
// CIR:         cir.scope {
// CIR:           %[[OUTER_ADDR:.*]] = cir.alloca !rec_StructWithDestructor, !cir.ptr<!rec_StructWithDestructor>, ["outer"]
// CIR:           cir.cleanup.scope {
// CIR:             cir.scope {
// CIR:               %[[INNER_ADDR:.*]] = cir.alloca !rec_StructWithDestructor, !cir.ptr<!rec_StructWithDestructor>, ["inner"]
// CIR:               cir.cleanup.scope {
// CIR:                 cir.scope {
// CIR:                   cir.if {{.*}} {
// CIR:                     cir.goto "end"
// CIR:                   }
// CIR:                 }
// CIR:                 cir.call @_Z3useR20StructWithDestructor(%[[INNER_ADDR]])
// CIR:                 cir.yield
// CIR:               } cleanup normal {
// CIR:                 cir.call @_ZN20StructWithDestructorD1Ev(%[[INNER_ADDR]])
// CIR:                 cir.yield
// CIR:               }
// CIR:             }
// CIR:             cir.call @_Z3useR20StructWithDestructor(%[[OUTER_ADDR]])
// CIR:             cir.yield
// CIR:           } cleanup normal {
// CIR:             cir.call @_ZN20StructWithDestructorD1Ev(%[[OUTER_ADDR]])
// CIR:             cir.yield
// CIR:           }
// CIR:         }
// CIR:         cir.br ^[[END_BB:bb[0-9]+]]
// CIR:       ^[[END_BB]]:
// CIR:         cir.label "end"
// CIR:         cir.return

// LLVM-LABEL: define dso_local void @_Z27test_goto_out_of_two_nestedb(i1 noundef %{{.*}})
// LLVM:         %[[OUTER_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         %[[INNER_ADDR:.*]] = alloca %struct.StructWithDestructor
// LLVM:         br i1 %{{.*}}, label %[[GOTO_PATH:.*]], label %[[NORMAL_PATH:.*]]
// LLVM:       [[GOTO_PATH]]:
// LLVM:         store i32 1, ptr %{{.*}}
// LLVM:         br label %[[INNER_CLEANUP:.*]]
// LLVM:       [[NORMAL_PATH]]:
// LLVM:         br label %[[USE_INNER:.*]]
// LLVM:       [[USE_INNER]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[INNER_ADDR]])
// LLVM:         store i32 0, ptr %{{.*}}
// LLVM:         br label %[[INNER_CLEANUP]]
// LLVM:       [[INNER_CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[INNER_ADDR]])
// LLVM:         switch i32 %{{.*}}, label %[[INNER_DEFAULT:.*]] [
// LLVM:           i32 0, label %[[INNER_FALL:.*]]
// LLVM:           i32 1, label %[[INNER_GOTO:.*]]
// LLVM:         ]
// LLVM:       [[INNER_FALL]]:
// LLVM:         br label %[[USE_OUTER:.*]]
// LLVM:       [[INNER_GOTO]]:
// LLVM:         store i32 1, ptr %{{.*}}
// LLVM:         br label %[[OUTER_CLEANUP:.*]]
// LLVM:       [[USE_OUTER]]:
// LLVM:         call void @_Z3useR20StructWithDestructor(ptr {{.*}} %[[OUTER_ADDR]])
// LLVM:         store i32 0, ptr %{{.*}}
// LLVM:         br label %[[OUTER_CLEANUP]]
// LLVM:       [[OUTER_CLEANUP]]:
// LLVM:         call void @_ZN20StructWithDestructorD1Ev(ptr {{.*}} %[[OUTER_ADDR]])
// LLVM-NOT:     call void @_ZN20StructWithDestructorD1Ev
// LLVM:         switch i32 %{{.*}}, label %[[OUTER_DEFAULT:.*]] [
// LLVM:           i32 0, label %[[OUTER_FALL:.*]]
// LLVM:           i32 1, label %[[OUTER_GOTO:.*]]
// LLVM:         ]
// LLVM:       [[OUTER_FALL]]:
// LLVM:         br label %[[END:.*]]
// LLVM:       [[END]]:
// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z27test_goto_out_of_two_nestedb(i1 noundef zeroext %{{.*}})
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
