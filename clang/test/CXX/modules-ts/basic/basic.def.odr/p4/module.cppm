// RUN: %clang_cc1 -fmodules-ts %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s --implicit-check-not=unused

static void unused_static_global_module() {}
static void used_static_global_module() {}
inline void unused_inline_global_module() {}
inline void used_inline_global_module() {}
// CHECK: define void {{.*}}@_Z23noninline_global_modulev
void noninline_global_module() {
  // FIXME: This should be promoted to module linkage and given a
  // module-mangled name, if it's called from an inline function within
  // the module interface.
  // (We should try to avoid this when it's not reachable from outside
  // the module interface unit.)
  // CHECK: define internal {{.*}}@_ZL25used_static_global_modulev
  used_static_global_module();
  // CHECK: define linkonce_odr {{.*}}@_Z25used_inline_global_modulev
  used_inline_global_module();
}

export module Module;

export {
  // FIXME: These should be ill-formed: you can't export an internal linkage
  // symbol, per [dcl.module.interface]p2.
  static void unused_static_exported() {}
  static void used_static_exported() {}

  inline void unused_inline_exported() {}
  inline void used_inline_exported() {}
  // CHECK: define void {{.*}}@_Z18noninline_exportedv
  void noninline_exported() {
    // CHECK: define internal {{.*}}@_ZL20used_static_exportedv
    used_static_exported();
    // CHECK: define linkonce_odr {{.*}}@_Z20used_inline_exportedv
    used_inline_exported();
  }
}

static void unused_static_module_linkage() {}
static void used_static_module_linkage() {}
inline void unused_inline_module_linkage() {}
inline void used_inline_module_linkage() {}
// FIXME: The module name should be mangled into the name of this function.
// CHECK: define void {{.*}}@_Z24noninline_module_linkagev
void noninline_module_linkage() {
  // FIXME: This should be promoted to module linkage and given a
  // module-mangled name, if it's called from an inline function within
  // the module interface.
  // CHECK: define internal {{.*}}@_ZL26used_static_module_linkagev
  used_static_module_linkage();
  // FIXME: The module name should be mangled into the name of this function.
  // CHECK: define linkonce_odr {{.*}}@_Z26used_inline_module_linkagev
  used_inline_module_linkage();
}
