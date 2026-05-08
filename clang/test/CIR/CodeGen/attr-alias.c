// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Variable alias to a defined target.
int alias_target = 42;
extern int alias_var __attribute__((alias("alias_target")));

// Function alias to a defined target.
void alias_func_target(void) {}
extern void alias_func(void) __attribute__((alias("alias_func_target")));

// Variable alias preceded by an extern declaration. The previous declaration
// must be replaced by the alias, otherwise we'd see two symbols with the same
// name.
int prior_decl_var_target = 7;
extern int prior_decl_var;
extern int prior_decl_var __attribute__((alias("prior_decl_var_target")));

// Function alias preceded by an extern declaration. Same as above but for the
// FuncOp entry-erasure path.
extern void prior_decl_func(void);
void prior_decl_func_target(void) {}
extern void prior_decl_func(void) __attribute__((alias("prior_decl_func_target")));

// Weak variable alias - exercises the WeakAttr linkage override branch.
int weak_var_target = 9;
extern int weak_var_alias __attribute__((weak, alias("weak_var_target")));

// Weak function alias - exercises the WeakAttr linkage override branch for
// functions.
void weak_func_target(void) {}
extern void weak_func_alias(void) __attribute__((weak, alias("weak_func_target")));

// CIR-DAG: cir.global external @alias_target = #cir.int<42> : !s32i
// CIR-DAG: cir.global external @alias_var alias(@alias_target) : !s32i
// CIR-DAG: cir.global external @prior_decl_var_target = #cir.int<7>
// CIR-DAG: cir.global external @prior_decl_var alias(@prior_decl_var_target) : !s32i
// CIR-DAG: cir.global external @weak_var_target = #cir.int<9>
// CIR-DAG: cir.global weak @weak_var_alias alias(@weak_var_target) : !s32i
// CIR-DAG: cir.func{{.*}} @alias_func_target()
// CIR-DAG: cir.func dso_local @alias_func() alias(@alias_func_target)
// CIR-DAG: cir.func{{.*}} @prior_decl_func_target()
// CIR-DAG: cir.func dso_local @prior_decl_func() alias(@prior_decl_func_target)
// CIR-DAG: cir.func{{.*}} @weak_func_target()
// CIR-DAG: cir.func weak @weak_func_alias() alias(@weak_func_target)
// CIR-DAG: cir.func dso_local @test13_alias(!u32i) alias(@test13)

// LLVM-DAG: @alias_target = global i32 42
// LLVM-DAG: @prior_decl_var_target = global i32 7
// LLVM-DAG: @weak_var_target = global i32 9
// LLVM-DAG: @alias_var = alias i32, ptr @alias_target
// LLVM-DAG: @prior_decl_var = alias i32, ptr @prior_decl_var_target
// LLVM-DAG: @weak_var_alias = weak alias i32, ptr @weak_var_target
// LLVM-DAG: @alias_func = alias void (), ptr @alias_func_target
// LLVM-DAG: @prior_decl_func = alias void (), ptr @prior_decl_func_target
// LLVM-DAG: @weak_func_alias = weak alias void (), ptr @weak_func_target
// LLVM-DAG: @test13_alias = alias void (i32), ptr @test13
// LLVM: define {{.*}}void @alias_func_target()

// OGCG-DAG: @alias_target = {{.*}}global i32 42
// OGCG-DAG: @prior_decl_var_target = {{.*}}global i32 7
// OGCG-DAG: @weak_var_target = {{.*}}global i32 9
// OGCG-DAG: @alias_var = {{.*}}alias i32, ptr @alias_target
// OGCG-DAG: @prior_decl_var = {{.*}}alias i32, ptr @prior_decl_var_target
// OGCG-DAG: @weak_var_alias = {{.*}}weak alias i32, ptr @weak_var_target
// OGCG-DAG: @alias_func = {{.*}}alias void (), ptr @alias_func_target
// OGCG-DAG: @prior_decl_func = {{.*}}alias void (), ptr @prior_decl_func_target
// OGCG-DAG: @weak_func_alias = {{.*}}weak alias void (), ptr @weak_func_target
// OGCG-DAG: @test13_alias = alias {}, ptr @test13
// OGCG: define {{.*}}void @alias_func_target()

// Test that a non visible (-Wvisibility) type doesn't assert.
enum a_type { test13_a };
void test13(enum a_type y) {}
void test13_alias(enum undeclared_type y) __attribute__((alias ("test13")));
