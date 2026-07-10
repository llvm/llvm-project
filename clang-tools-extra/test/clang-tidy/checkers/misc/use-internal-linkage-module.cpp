// RUN: %check_clang_tidy -std=c++20-or-later %s misc-use-internal-linkage %t

export module test;

export void single_export_fn() {}
export int single_export_var;
export struct SingleExportStruct {};

export {
  void group_export_fn1() {}
  void group_export_fn2() {}
  int group_export_var1;
  int group_export_var2;
  struct GroupExportStruct1 {};
  struct GroupExportStruct2 {};
}

export namespace aa {
void namespace_export_fn() {}
int namespace_export_var;
struct NamespaceExportStruct {};
} // namespace aa

void unexported_fn() {}
int unexported_var;
struct UnexportedStruct {};

module : private;

void fn_in_private_module_fragment() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'fn_in_private_module_fragment' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static void fn_in_private_module_fragment() {}
int var_in_private_module_fragment;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'var_in_private_module_fragment' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static int var_in_private_module_fragment;
struct StructInPrivateModuleFragment {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: struct 'StructInPrivateModuleFragment' can be moved into an anonymous namespace to enforce internal linkage
