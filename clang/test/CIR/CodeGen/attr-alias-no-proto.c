// RUN: %clang_cc1 -std=c11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c11 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c11 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// LLVM: @noproto_used = alias i32 (...), ptr @noproto_used_target
// LLVM: @noproto_args = alias i32 (...), ptr @noproto_args_target
// LLVM: @noproto_args2 = alias i32 (i32, i32, i32), ptr @noproto_args_target2

// OGCG: @noproto_used = alias i32 (...), ptr @noproto_used_target
// OGCG: @noproto_args = alias i32 (...), ptr @noproto_args_target
// OGCG: @noproto_args2 = alias i32 (i32, i32, i32), ptr @noproto_args_target2

// Use of a no-prototype function, then alias definition sets the type.
extern int noproto_used();
int noproto_use_it(void) { return noproto_used(); }
int noproto_used_target(void) { return 0; }
int noproto_used() __attribute__((alias("noproto_used_target")));

// Note: Function attrs no_inline/dso_local included to show that no_proto is
// NOT present.
// CIR-LABEL: cir.func no_inline dso_local @noproto_use_it() -> !s32i
// CIR:  %[[GET_USED:.*]] = cir.get_global @noproto_used : !cir.ptr<!cir.func<() -> !s32i>>
// CIR:  cir.call %[[GET_USED]]() : (!cir.ptr<!cir.func<() -> !s32i>>) -> !s32i

// LLVM-LABEL: define dso_local i32 @noproto_use_it()
// LLVM: call i32 (...) @noproto_used()
// OGCG: define dso_local i32 @noproto_use_it()
// OGCG: call i32 (...) @noproto_used()

// CIR-LABEL: cir.func no_inline dso_local @noproto_used_target() -> !s32i
// LLVM-LABEL: define dso_local i32 @noproto_used_target()
// OGCG-LABEL: define dso_local i32 @noproto_used_target()

// CIR: cir.func no_proto dso_local @noproto_used() -> !s32i alias(@noproto_used_target)
// LLVM sorts these at the top of the file.

// Use of a no-prototype function with args, then alias definition sets the type.
extern int noproto_args();
int noproto_args_use(void) { return noproto_args(1, 2, 3); }
int noproto_args_target(void) { return 0; }
int noproto_args() __attribute__((alias("noproto_args_target")));
// CIR-LABEL: cir.func no_inline dso_local @noproto_args_use() -> !s32i
// CIR: %[[GET_NPA:.*]] = cir.get_global @noproto_args : !cir.ptr<!cir.func<() -> !s32i>>
// CIR: %[[TO_VARIADIC:.*]] = cir.cast bitcast %[[GET_NPA]] : !cir.ptr<!cir.func<() -> !s32i>> -> !cir.ptr<!cir.func<(...) -> !s32i>>
// CIR: %[[TO_TYPED:.*]] = cir.cast bitcast %[[TO_VARIADIC]] : !cir.ptr<!cir.func<(...) -> !s32i>> -> !cir.ptr<!cir.func<(!s32i, !s32i, !s32i) -> !s32i>>
// CIR: cir.call %[[TO_TYPED]](%{{.*}}, %{{.*}}, %{{.*}}) : (!cir.ptr<!cir.func<(!s32i, !s32i, !s32i) -> !s32i>>, !s32i {llvm.noundef}, !s32i {llvm.noundef}, !s32i {llvm.noundef}) -> !s32i
//
// LLVM-LABEL: define dso_local i32 @noproto_args_use()
// LLVM: call i32 (i32, i32, i32, ...) @noproto_args(i32 noundef 1, i32 noundef 2, i32 noundef 3)
// OGCG-LABEL: define dso_local i32 @noproto_args_use()
// OGCG: call i32 (i32, i32, i32, ...) @noproto_args(i32 noundef 1, i32 noundef 2, i32 noundef 3)

// CIR-LABEL: cir.func no_inline dso_local @noproto_args_target() -> !s32i
// LLVM-LABEL: define dso_local i32 @noproto_args_target()
// OGCG-LABEL: define dso_local i32 @noproto_args_target()

// CIR:  cir.func no_proto dso_local @noproto_args() -> !s32i alias(@noproto_args_target)
// LLVM sorts these at the top of the file.

// Use of a no-prototype function with args, alias adds args, type fixed before
// use, so casts unnecessary.
extern int noproto_args2();
int noproto_args_use2(void) { return noproto_args2(1, 2, 3); }
int noproto_args_target2(void) { return 0; }
int noproto_args2(int, int, int) __attribute__((alias("noproto_args_target2")));

// CIR-LABEL: cir.func no_inline dso_local @noproto_args_use2() -> !s32i
// CIR: %[[GET_NPA:.*]] = cir.get_global @noproto_args2 : !cir.ptr<!cir.func<(!s32i, !s32i, !s32i) -> !s32i>>
// CIR: cir.call %4(%{{.*}}, %{{.*}}, %{{.*}}) : (!cir.ptr<!cir.func<(!s32i, !s32i, !s32i) -> !s32i>>, !s32i {llvm.noundef}, !s32i {llvm.noundef}, !s32i {llvm.noundef}) -> !s32i
//
// LLVM-LABEL: define dso_local i32 @noproto_args_use2()
// LLVM: call i32 (i32, i32, i32, ...) @noproto_args2(i32 noundef 1, i32 noundef 2, i32 noundef 3)
// OGCG-LABEL: define dso_local i32 @noproto_args_use2()
// OGCG: call i32 (i32, i32, i32, ...) @noproto_args2(i32 noundef 1, i32 noundef 2, i32 noundef 3)

// CIR-LABEL: cir.func no_inline dso_local @noproto_args_target2() -> !s32i
// LLVM-LABEL: define dso_local i32 @noproto_args_target2()
// OGCG-LABEL: define dso_local i32 @noproto_args_target2()

// CIR:  cir.func no_proto dso_local @noproto_args2(!s32i, !s32i, !s32i) -> !s32i alias(@noproto_args_target2)
// LLVM sorts these at the top of the file.
