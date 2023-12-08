// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -O2 -fsanitize=kernel-memory -no-enable-noundef-analysis -o - %s | \
// RUN:     FileCheck %s --check-prefix=CLEAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -O2 -fsanitize=kernel-memory -fno-sanitize-memory-param-retval -o - %s | \
// RUN:     FileCheck %s --check-prefixes=NOUNDEF,NOUNDEF_ONLY
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -O2 -fsanitize=kernel-memory -mllvm -msan-eager-checks -o - %s | \
// RUN:     FileCheck %s --check-prefixes=NOUNDEF,EAGER
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -O2 -fsanitize=kernel-memory -no-enable-noundef-analysis -fsanitize-memory-param-retval -o - %s | \
// RUN:     FileCheck %s --check-prefixes=CLEAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -O2 -fsanitize=kernel-memory -o - %s | \
// RUN:     FileCheck %s --check-prefixes=NOUNDEF,EAGER

void foo();

void bar(int x) {
  if (x)
    foo();
}


// CLEAN:   define dso_local void @bar(i32 %x)
// NOUNDEF: define dso_local void @bar(i32 noundef %x)
//
// %param_shadow assignment gets optimized away with -O2, because it is at the beginning of the
// struct returned by __msan_get_context_state(). Use %param_origin as a sign that the shadow of
// the first argument is being used.
//
// Without noundef analysis, KMSAN emits metadata checks for the function parameter.
// CLEAN:        load i32, ptr %param_origin
//
// With noundef analysis enabled, but without eager checks, KMSAN still emits metadata checks,
// although the parameter is known to be defined.
// NOUNDEF_ONLY: load i32, ptr %param_origin
//
// With noundef analysis and eager checks enabled, KMSAN won't emit metadata checks for function
// parameters.
// EAGER-NOT:    load i32, ptr %param_origin
