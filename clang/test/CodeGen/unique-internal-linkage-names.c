// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -funique-internal-linkage-names -o - | FileCheck %s

// CHECK: @_ZL{{[0-9]+}}aliased_var_target.[[MODHASH:__uniq.[0-9]+]] = internal constant ptr
// CHECK: @_ZL{{[0-9]+}}late_aliased_var_target.[[MODHASH]] = internal global i32 1
// CHECK: @aliased_var ={{.*}} alias ptr, ptr @_ZL{{[0-9]+}}aliased_var_target.[[MODHASH]]
// CHECK: @late_aliased_var ={{.*}} alias i32, ptr @_ZL{{[0-9]+}}late_aliased_var_target.[[MODHASH]]

// Check that we do not crash when overloading extern functions.

inline void overloaded_external() {}
extern void overloaded_external();

// A prototyped static function gets a unique suffix...
// CHECK: define internal i32 @_ZL7uniquedv.__uniq.{{[0-9]+}}(
static int uniqued(void) { return 0; }

// Check that a static function with asm label keeps its original name.
// CHECK: define internal i32 @custom_label(
static int asm_label(void) asm("custom_label");
static int asm_label(void) { return 0; }

// An alias can name an internal variable without knowing its unique suffix.
static const void *const aliased_var_target = &aliased_var_target;
extern void *aliased_var __attribute__((alias("aliased_var_target")));

// The target may also be declared after its alias.
extern int late_aliased_var
    __attribute__((alias("late_aliased_var_target")));
static int late_aliased_var_target = 1;

// CHECK: define internal void @overloaded_internal() [[ATTR:#[0-9]+]] {
static void overloaded_internal() {}
extern void overloaded_internal();

void markUsed() {
  overloaded_external();
  overloaded_internal();
  uniqued();
  asm_label();
}

// CHECK: attributes [[ATTR]] =
// CHECK-SAME: "sample-profile-suffix-elision-policy"="selected"
