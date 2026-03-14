// RUN: %clang_cc1 -std=gnu2x -triple x86_64-linux-gnu %s -emit-llvm -o - \
// RUN:   -Werror=ignored-attributes \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-NOM
// RUN: %clang_cc1 -std=gnu2x -triple x86_64-linux-gnu %s -emit-llvm -o - \
// RUN:   -Werror=ignored-attributes -mfunction-return=keep \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-KEEP
// RUN: %clang_cc1 -std=gnu2x -triple x86_64-linux-gnu %s -emit-llvm -o - \
// RUN:   -Werror=ignored-attributes -mfunction-return=thunk-extern \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-EXTERN
// RUN: %clang_cc1 -std=gnu2x -triple x86_64-linux-gnu %s -emit-llvm -o - \
// RUN:  -mfunction-return=thunk-extern -coverage-data-file=/dev/null \
// RUN:   | FileCheck %s --check-prefix=CHECK-GCOV
// RUN: %clang_cc1 -std=gnu2x -triple x86_64-linux-gnu %s -emit-llvm -o - \
// RUN:  -mfunction-return=thunk-extern -fsanitize=address \
// RUN:   | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang_cc1 -std=gnu2x -triple x86_64-linux-gnu %s -emit-llvm -o - \
// RUN:  -mfunction-return=thunk-extern -fsanitize=thread \
// RUN:   | FileCheck %s --check-prefix=CHECK-TSAN

#if !__has_attribute(function_return)
#error "missing attribute support for function_return"
#endif

// CHECK: @keep() [[KEEP:#[0-9]+]]
__attribute__((function_return("keep"))) void keep(void) {}

// CHECK: @keep2() [[KEEP]]
[[gnu::function_return("keep")]] void keep2(void) {}

// CHECK: @thunk_extern() [[EXTERN:#[0-9]+]]
__attribute__((function_return("thunk-extern"))) void thunk_extern(void) {}

// CHECK: @thunk_extern2() [[EXTERN]]
[[gnu::function_return("thunk-extern")]] void thunk_extern2(void) {}

// CHECK: @double_thunk_keep() [[KEEP]]
// clang-format off
__attribute__((function_return("thunk-extern")))
__attribute__((function_return("keep")))
void double_thunk_keep(void) {}

// CHECK: @double_thunk_keep2() [[KEEP]]
[[gnu::function_return("thunk-extern")]][[gnu::function_return("keep")]]
void double_thunk_keep2(void) {}

// CHECK: @double_keep_thunk() [[EXTERN]]
__attribute__((function_return("keep")))
__attribute__((function_return("thunk-extern")))
void double_keep_thunk(void) {}

// CHECK: @double_keep_thunk2() [[EXTERN]]
[[gnu::function_return("keep")]][[gnu::function_return("thunk-extern")]]
void double_keep_thunk2(void) {}

// CHECK: @thunk_keep() [[KEEP]]
__attribute__((function_return("thunk-extern"), function_return("keep")))
void thunk_keep(void) {}

// CHECK: @thunk_keep2() [[KEEP]]
[[gnu::function_return("thunk-extern"), gnu::function_return("keep")]]
void thunk_keep2(void) {}

// CHECK: @keep_thunk() [[EXTERN]]
__attribute__((function_return("keep"), function_return("thunk-extern")))
void keep_thunk(void) {}

// CHECK: @keep_thunk2() [[EXTERN]]
[[gnu::function_return("keep"), gnu::function_return("thunk-extern")]]
void keep_thunk2(void) {}
// clang-format on

void undef(void);
// CHECK: @undef() [[KEEP]]
__attribute__((function_return("keep"))) void undef(void) {}

void undef2(void);
// CHECK: @undef2() [[EXTERN]]
__attribute__((function_return("thunk-extern"))) void undef2(void) {}

__attribute__((function_return("thunk-extern"))) void change_def(void);
// CHECK: @change_def() [[KEEP]]
__attribute__((function_return("keep"))) void change_def(void) {}

__attribute__((function_return("keep"))) void change_def2(void);
// CHECK: @change_def2() [[EXTERN]]
__attribute__((function_return("thunk-extern"))) void change_def2(void) {}

__attribute__((function_return("thunk-extern"))) void change_def3(void);
// CHECK: @change_def3() [[KEEP]]
[[gnu::function_return("keep")]] void change_def3(void) {}

[[gnu::function_return("keep")]] void change_def4(void);
// CHECK: @change_def4() [[EXTERN]]
__attribute__((function_return("thunk-extern"))) void change_def4(void) {}

// When there is no -mfunction-return= flag set (NOM) or it's set to keep,
// we don't emit anything into the IR for unattributed functions.

// CHECK-NOM:    @no_attrs() [[NOATTR:#[0-9]+]]
// CHECK-KEEP:   @no_attrs() [[NOATTR:#[0-9]+]]
// CHECK-EXTERN: @no_attrs() [[EXTERN]]
void no_attrs(void) {}

// Test synthetic functions.
// CHECK-GCOV: @__llvm_gcov_writeout() unnamed_addr [[EXTERNGCOV:#[0-9]+]]
// CHECK-GCOV: @__llvm_gcov_reset() unnamed_addr [[EXTERNGCOV]]
// CHECK-GCOV: @__llvm_gcov_init() unnamed_addr [[EXTERNGCOV]]
// CHECK-ASAN: @asan.module_ctor() [[EXTERNASAN:#[0-9]+]]
// CHECK-TSAN: @tsan.module_ctor() [[EXTERNTSAN:#[0-9]+]]

// CHECK-NOM-NOT:  [[NOATTR]] = {{.*}}fn_ret_thunk_extern
// CHECK-KEEP-NOT: [[NOATTR]] = {{.*}}fn_ret_thunk_extern
// CHECK: [[EXTERN]] = {{.*}}fn_ret_thunk_extern
// CHECK-GCOV: [[EXTERNGCOV]] = {{.*}}fn_ret_thunk_extern
// CHECK-ASAN: [[EXTERNASAN]] = {{.*}}fn_ret_thunk_extern
// CHECK-TSAN: [[EXTERNTSAN]] = {{.*}}fn_ret_thunk_extern
