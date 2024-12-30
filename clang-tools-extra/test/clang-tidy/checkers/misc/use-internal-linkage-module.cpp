// RUN: %check_clang_tidy -std=c++20 %s misc-use-internal-linkage %t -- -- -I%S/Inputs/use-internal-linkage

module;

export module test;

export void single_export_fn() {}
export int single_export_var;

export {
  void group_export_fn1() {}
  void group_export_fn2() {}
  int group_export_var1;
  int group_export_var2;
}

export namespace aa {
void namespace_export_fn() {}
int namespace_export_var;
} // namespace aa
