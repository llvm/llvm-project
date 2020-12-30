// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -mrelocation-model pic -pic-level 1 -fno-semantic-interposition %s -o - | FileCheck %s

/// For ELF -fpic/-fPIC, if -fno-semantic-interposition is specified, mark
/// defined variables and functions dso_local. ifunc isn't marked.

// CHECK: @var = dso_local global i32 0, align 4
// CHECK: @ext_var = external global i32, align 4
int var;
extern int ext_var;

// CHECK: @ifunc = ifunc i32 (), bitcast (i8* ()* @ifunc_resolver to i32 ()*)
int ifunc(void) __attribute__((ifunc("ifunc_resolver")));

// CHECK: define dso_local i32 @func()
// CHECK: declare i32 @ext()
int func(void) { return 0; }
int ext(void);

static void *ifunc_resolver() { return func; }

int foo() {
  return var + ext_var + ifunc() + func() + ext();
}
