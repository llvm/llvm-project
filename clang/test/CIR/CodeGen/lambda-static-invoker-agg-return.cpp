// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct S {
  int x;
  S();
  S(const S &);
  S &operator=(const S &);
};

S make_s();

S agg_invoker() {
  auto *fn = +[](int i) -> S { return make_s(); };
  return fn(3);
}

// CIR-LABEL: cir.func no_inline internal private dso_local @_ZZ11agg_invokervEN3$_08__invokeEi
// CIR-SAME:    (%[[I_ARG:.*]]: !s32i {{.*}}) -> !rec_S
// CIR:         %[[I_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR:         %[[RETVAL:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["__retval"]
// CIR:         %[[UNUSED:.*]] = cir.alloca !rec_anon{{[^,]*}}, {{.*}} ["unused.capture"]
// CIR-NOT:     cir.alloca {{.*}} ["agg.tmp
// CIR:         cir.store %[[I_ARG]], %[[I_ALLOCA]]
// CIR:         %[[I:.*]] = cir.load{{.*}} %[[I_ALLOCA]]
// CIR:         %[[CALL:.*]] = cir.call @_ZZ11agg_invokervENK3$_0clEi(%[[UNUSED]], %[[I]]){{.*}} -> !rec_S
// CIR-NOT:     cir.copy
// CIR:         cir.store{{.*}} %[[CALL]], %[[RETVAL]]
// CIR:         %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR:         cir.return %[[RET]]

// LLVM-LABEL: define internal %struct.S @"_ZZ11agg_invokervEN3$_08__invokeEi"
// LLVM-SAME:    (i32 {{[^,)]*}} %[[I_ARG:[^,)]+]])
// LLVM:         %[[I_ALLOCA:.*]] = alloca i32
// LLVM:         %[[RETVAL:.*]] = alloca %struct.S
// LLVM:         %[[UNUSED:.*]] = alloca %class.anon
// LLVM:         store i32 %[[I_ARG]], ptr %[[I_ALLOCA]]
// LLVM:         %[[I:.*]] = load i32, ptr %[[I_ALLOCA]]
// LLVM:         %[[CALL:.*]] = call %struct.S @"_ZZ11agg_invokervENK3$_0clEi"(ptr {{.*}} %[[UNUSED]], i32 {{.*}} %[[I]])
// LLVM:         store %struct.S %[[CALL]], ptr %[[RETVAL]]
// LLVM:         %[[RET:.*]] = load %struct.S, ptr %[[RETVAL]]
// LLVM:         ret %struct.S %[[RET]]

// OGCG-LABEL: define internal void @"_ZZ11agg_invokervEN3$_08__invokeEi"
// OGCG-SAME:    (ptr {{.*}} sret(%struct.S) {{[^,]*}} %[[AGG_RESULT:[^,]+]], i32 {{[^,)]*}} %[[I_ARG:[^,)]+]])
// OGCG:         call void @"_ZZ11agg_invokervENK3$_0clEi"(ptr {{.*}} sret(%struct.S) {{[^,]*}} %[[AGG_RESULT]], ptr {{.*}} %[[UNUSED:.*]], i32 {{.*}})
// OGCG:         ret void
