/// -fno-semantic-interposition is the default and local aliases (via dso_local) are allowed.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -mrelocation-model pic -pic-level 1 %s -o - | FileCheck %s --check-prefixes=CHECK,NOMETADATA

/// -fsemantic-interposition sets a module metadata.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -mrelocation-model pic -pic-level 1 -fsemantic-interposition %s -o - | FileCheck %s --check-prefixes=PREEMPT,METADATA

/// Traditional half-baked behavior: interprocedural optimizations are allowed
/// but local aliases are not used.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -mrelocation-model pic -pic-level 1 -fhalf-no-semantic-interposition %s -o - | FileCheck %s --check-prefixes=PREEMPT,NOMETADATA

// CHECK: @var = global i32 0, align 4
// CHECK: @ext_var = external global i32, align 4
// CHECK: @ifunc = ifunc i32 (), ptr @ifunc_resolver
// CHECK: define dso_local i32 @func()
// CHECK: declare i32 @ext()

// PREEMPT: @var = global i32 0, align 4
// PREEMPT: @ext_var = external global i32, align 4
// PREEMPT: @ifunc = ifunc i32 (), ptr @ifunc_resolver
// PREEMPT: define i32 @func()
// PREEMPT: declare i32 @ext()

// METADATA:           !{{[0-9]+}} = !{i32 1, !"SemanticInterposition", i32 1}
// NOMETADATA-NOT:     "SemanticInterposition"

int var;
extern int ext_var;

int ifunc(void) __attribute__((ifunc("ifunc_resolver")));

int func(void) { return 0; }
int ext(void);

static void *ifunc_resolver(void) { return func; }

int foo(void) {
  return var + ext_var + ifunc() + func() + ext();
}
