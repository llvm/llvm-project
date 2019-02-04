// RUN: %clang_cc1 %s -std=c11 -triple=x86_64-pc-linux -fvisibility hidden -fapply-global-visibility-to-externs -emit-llvm -o - | FileCheck --check-prefix=CHECK-HIDDEN %s
// RUN: %clang_cc1 %s -std=c11 -triple=x86_64-pc-linux -fvisibility protected -fapply-global-visibility-to-externs -emit-llvm -o - | FileCheck --check-prefix=CHECK-PROTECTED %s
// RUN: %clang_cc1 %s -std=c11 -triple=x86_64-pc-linux -fvisibility default -fapply-global-visibility-to-externs -emit-llvm -o - | FileCheck --check-prefix=CHECK-DEFAULT %s

// CHECK-HIDDEN: @var_hidden = external hidden global
// CHECK-PROTECTED: @var_hidden = external hidden global
// CHECK-DEFAULT: @var_hidden = external hidden global
__attribute__((visibility("hidden"))) extern int var_hidden;
// CHECK-HIDDEN: @var_protected = external protected global
// CHECK-PROTECTED: @var_protected = external protected global
// CHECK-DEFAULT: @var_protected = external protected global
__attribute__((visibility("protected"))) extern int var_protected;
// CHECK-HIDDEN: @var_default = external global
// CHECK-PROTECTED: @var_default = external global
// CHECK-DEFAULT: @var_default = external global
__attribute__((visibility("default"))) extern int var_default;
// CHECK-HIDDEN: @var = external hidden global
// CHECK-PROTECTED: @var = external protected global
// CHECK-DEFAULT: @var = external global
extern int var;

// CHECK-HIDDEN: declare hidden i32 @func_hidden()
// CHECK-PROTECTED: declare hidden i32 @func_hidden()
// CHECK-DEFAULT: declare hidden i32 @func_hidden()
__attribute__((visibility("hidden"))) int func_hidden(void);
// CHECK-HIDDEN: declare protected i32 @func_protected()
// CHECK-PROTECTED: declare protected i32 @func_protected()
// CHECK-DEFAULT: declare protected i32 @func_protected()
__attribute__((visibility("protected"))) int func_protected(void);
// CHECK-HIDDEN: declare i32 @func_default()
// CHECK-PROTECTED: declare i32 @func_default()
// CHECK-DEFAULT: declare i32 @func_default()
__attribute__((visibility("default"))) int func_default(void);
// CHECK-HIDDEN: declare hidden i32 @func()
// CHECK-PROTECTED: declare protected i32 @func()
// CHECK-DEFAULT: declare i32 @func()
int func(void);

int use() {
  return var_hidden + var_protected + var_default + var +
         func_hidden() + func_protected() + func_default() + func();
}
