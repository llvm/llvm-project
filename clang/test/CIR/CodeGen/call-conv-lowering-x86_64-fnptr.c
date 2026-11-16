// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: FileCheck --check-prefix=CIRGLOBAL --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: FileCheck --check-prefix=DECL --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=DECL --input-file=%t.ll %s

// Module-level items, which the two backends order differently against the
// function definitions, so DECL has no label blocks to be constrained by.
// DECL-DAG: declare void @make_big(ptr dead_on_unwind writable sret(%struct.Big) align 8, i32 noundef)
// DECL-DAG: declare i64 @make_pair(i32 noundef)
// DECL-DAG: @g_big = global ptr @make_big, align 8
// DECL-DAG: @g_arr = global [1 x ptr] [ptr @make_big], align 8
// DECL-DAG: @g_rec = global %struct.HolderS { ptr @make_big }, align 8

typedef struct { long a, b, c, d; } Big;
typedef struct { int x, y; } Pair2;

Big make_big(int);
Pair2 make_pair(int);

typedef Big (*BigFP)(int);
typedef Pair2 (*PairFP)(int);

// Global initializers hold a GlobalViewAttr, not a cir.get_global.
BigFP g_big = make_big;
BigFP g_arr[1] = { make_big };
struct HolderS { BigFP p; };
struct HolderS g_rec = { make_big };

// CIRGLOBAL-DAG: cir.global external @g_big = #cir.global_view<@make_big> : !cir.ptr<!cir.func<(!s32i) -> !rec_Big>>
// CIRGLOBAL-DAG: cir.global external @g_arr = #cir.const_array<[#cir.global_view<@make_big> : !cir.ptr<!cir.func<(!s32i) -> !rec_Big>>]>
// CIRGLOBAL-DAG: cir.global external @g_rec = #cir.const_record<{#cir.global_view<@make_big> : !cir.ptr<!cir.func<(!s32i) -> !rec_Big>>}>

// A 32-byte return becomes an sret parameter.
BigFP get_big(void) { return make_big; }

// CIR-LABEL: cir.func{{.*}} @get_big() -> !cir.ptr<!cir.func<(!s32i) -> !rec_Big>>
// CIR:         %[[G:[0-9]+]] = cir.get_global @make_big : !cir.ptr<!cir.func<(!cir.ptr<!rec_Big>, !s32i)>>
// CIR-NEXT:    %[[C:[0-9]+]] = cir.cast bitcast %[[G]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_Big>, !s32i)>> -> !cir.ptr<!cir.func<(!s32i) -> !rec_Big>>
// CIR-NEXT:    cir.store %[[C]], %{{[0-9]+}} :

// LLVM-LABEL: define dso_local ptr @get_big()
// LLVM-CIR:     store ptr @make_big, ptr %[[SLOT:[a-zA-Z0-9._]+]], align 8
// LLVM-CIR:     %[[FP:[a-zA-Z0-9._]+]] = load ptr, ptr %[[SLOT]], align 8
// LLVM-CIR:     ret ptr %[[FP]]
// LLVM-OGCG:    ret ptr @make_big{{$}}

// An 8-byte return is coerced into a register, changing the return type.
PairFP get_pair(void) { return make_pair; }

// CIR-LABEL: cir.func{{.*}} @get_pair() -> !cir.ptr<!cir.func<(!s32i) -> !rec_Pair2>>
// CIR:         %[[G:[0-9]+]] = cir.get_global @make_pair : !cir.ptr<!cir.func<(!s32i) -> !u64i>>
// CIR-NEXT:    %[[C:[0-9]+]] = cir.cast bitcast %[[G]] : !cir.ptr<!cir.func<(!s32i) -> !u64i>> -> !cir.ptr<!cir.func<(!s32i) -> !rec_Pair2>>

// LLVM-LABEL: define dso_local ptr @get_pair()
// LLVM-CIR:     store ptr @make_pair, ptr %{{[a-zA-Z0-9._]+}}, align 8
// LLVM-OGCG:    ret ptr @make_pair{{$}}

// The address of a definition, whose body the sret rewrite also rewires.
Big make_big_def(int n) { Big b = {n, n, n, n}; return b; }
BigFP get_def(void) { return make_big_def; }

// CIR-LABEL: cir.func{{.*}} @make_big_def(%arg0: !cir.ptr<!rec_Big> {{{.*}}llvm.sret = !rec_Big{{.*}}, %arg1: !s32i {llvm.noundef}
// CIR-LABEL: cir.func{{.*}} @get_def() -> !cir.ptr<!cir.func<(!s32i) -> !rec_Big>>
// CIR:         %[[G:[0-9]+]] = cir.get_global @make_big_def : !cir.ptr<!cir.func<(!cir.ptr<!rec_Big>, !s32i)>>
// CIR-NEXT:    %[[C:[0-9]+]] = cir.cast bitcast %[[G]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_Big>, !s32i)>> -> !cir.ptr<!cir.func<(!s32i) -> !rec_Big>>

// LLVM-LABEL: define dso_local void @make_big_def(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{.+}}, i32 noundef %{{.+}})
// LLVM-LABEL: define dso_local ptr @get_def()
// LLVM-CIR:     store ptr @make_big_def, ptr %{{[a-zA-Z0-9._]+}}, align 8
// LLVM-OGCG:    ret ptr @make_big_def{{$}}

// Calling through the address reaches the callee at its rewritten signature.
Big call_local(int n) {
  BigFP f = make_big;
  return f(n);
}

// CIR-LABEL: cir.func{{.*}} @call_local(%arg0: !cir.ptr<!rec_Big> {{{.*}}llvm.sret = !rec_Big{{.*}}, %arg1: !s32i {llvm.noundef}
// CIR:         %[[G:[0-9]+]] = cir.get_global @make_big : !cir.ptr<!cir.func<(!cir.ptr<!rec_Big>, !s32i)>>
// CIR-NEXT:    %[[C:[0-9]+]] = cir.cast bitcast %[[G]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_Big>, !s32i)>> -> !cir.ptr<!cir.func<(!s32i) -> !rec_Big>>
// CIR-NEXT:    cir.store align(8) %[[C]], %[[SLOT:[0-9]+]] :
// CIR-NEXT:    %[[LOADED:[0-9]+]] = cir.load align(8) %[[SLOT]] :
// CIR:         %[[BACK:[0-9]+]] = cir.cast bitcast %[[LOADED]] : !cir.ptr<!cir.func<(!s32i) -> !rec_Big>> -> !cir.ptr<!cir.func<(!cir.ptr<!rec_Big>, !s32i)>>
// CIR-NEXT:    cir.call %[[BACK]](%arg0, %{{[0-9]+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Big>, !s32i)>>, !cir.ptr<!rec_Big> {{{.*}}llvm.sret = !rec_Big{{.*}}, !s32i {llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @call_local(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{.+}}, i32 noundef %{{.+}})
// LLVM:         store ptr @make_big, ptr %[[SLOT:[a-zA-Z0-9._]+]], align 8
// LLVM:         %[[FP:[a-zA-Z0-9._]+]] = load ptr, ptr %[[SLOT]], align 8
// LLVM:         call void %[[FP]](ptr dead_on_unwind writable sret(%struct.Big) align 8 %{{.+}}, i32 noundef %{{.+}})
