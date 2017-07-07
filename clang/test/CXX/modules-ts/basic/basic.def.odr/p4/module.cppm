// RUN: %clang_cc1 -fmodules-ts %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s --implicit-check-not unused_inline --implicit-check-not unused_stastic_global_module

// CHECK-DAG: @extern_var_global_module = external global
// CHECK-DAG: @inline_var_global_module = linkonce_odr global
// CHECK-DAG: @_ZL24static_var_global_module = internal global
// CHECK-DAG: @_ZL23const_var_global_module = internal constant
//
// For ABI compatibility, these symbols do not include the module name.
// CHECK-DAG: @extern_var_exported = external global
// FIXME: Should this be 'weak_odr global'? Presumably it must be, since we
// can discard this global and its initializer (if any), and other TUs are not
// permitted to run the initializer for this variable.
// CHECK-DAG: @inline_var_exported = linkonce_odr global
// CHECK-DAG: @_ZL19static_var_exported = global
// CHECK-DAG: @const_var_exported = constant
//
// FIXME: The module name should be mangled into all of these.
// CHECK-DAG: @extern_var_module_linkage = external global
// FIXME: Should this be 'weak_odr global'? Presumably it must be, since we
// can discard this global and its initializer (if any), and other TUs are not
// permitted to run the initializer for this variable.
// CHECK-DAG: @inline_var_module_linkage = linkonce_odr global
// CHECK-DAG: @_ZL25static_var_module_linkage = global
// CHECK-DAG: @_ZL24const_var_module_linkage = constant

static void unused_static_global_module() {}
static void used_static_global_module() {}

inline void unused_inline_global_module() {}
inline void used_inline_global_module() {}

extern int extern_var_global_module;
inline int inline_var_global_module;
static int static_var_global_module;
const int const_var_global_module = 3;

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

  (void)&extern_var_global_module;
  (void)&inline_var_global_module;
  (void)&static_var_global_module;
  (void)&const_var_global_module;
}

export module Module;

export {
  // FIXME: These should be ill-formed: you can't export an internal linkage
  // symbol, per [dcl.module.interface]p2.
  // CHECK: define void {{.*}}@_ZL22unused_static_exportedv
  static void unused_static_exported() {}
  // CHECK: define void {{.*}}@_ZL20used_static_exportedv
  static void used_static_exported() {}

  inline void unused_inline_exported() {}
  inline void used_inline_exported() {}

  extern int extern_var_exported;
  inline int inline_var_exported;
  // FIXME: This should be ill-formed: you can't export an internal linkage
  // symbol.
  static int static_var_exported;
  const int const_var_exported = 3;

  // CHECK: define void {{.*}}@_Z18noninline_exportedv
  void noninline_exported() {
    used_static_exported();
    // CHECK: define linkonce_odr {{.*}}@_Z20used_inline_exportedv
    used_inline_exported();

    (void)&extern_var_exported;
    (void)&inline_var_exported;
    (void)&static_var_exported;
    (void)&const_var_exported;
  }
}

// FIXME: Ideally we wouldn't emit this as its name is not visible outside this
// TU, but this module interface might contain a template that can use this
// function so we conservatively emit it for now.
// FIXME: The module name should be mangled into the name of this function.
// CHECK: define void {{.*}}@_ZL28unused_static_module_linkagev
static void unused_static_module_linkage() {}
// FIXME: The module name should be mangled into the name of this function.
// CHECK: define void {{.*}}@_ZL26used_static_module_linkagev
static void used_static_module_linkage() {}

inline void unused_inline_module_linkage() {}
inline void used_inline_module_linkage() {}

extern int extern_var_module_linkage;
inline int inline_var_module_linkage;
static int static_var_module_linkage;
const int const_var_module_linkage = 3;

// FIXME: The module name should be mangled into the name of this function.
// CHECK: define void {{.*}}@_Z24noninline_module_linkagev
void noninline_module_linkage() {
  used_static_module_linkage();
  // FIXME: The module name should be mangled into the name of this function.
  // CHECK: define linkonce_odr {{.*}}@_Z26used_inline_module_linkagev
  used_inline_module_linkage();

  (void)&extern_var_module_linkage;
  (void)&inline_var_module_linkage;
  (void)&static_var_module_linkage;
  (void)&const_var_module_linkage;
}
