// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Module.cppm -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %t/Module.cppm --implicit-check-not unused
//
// RUN: %clang_cc1 -std=c++20 %t/Module.cppm -triple %itanium_abi_triple -emit-module-interface -o %t/Module.pcm
// RUN: %clang_cc1 -std=c++20 %t/module.cpp -triple %itanium_abi_triple -fmodule-file=%t/Module.pcm -emit-llvm -o - | FileCheck %t/module.cpp --implicit-check-not=unused --implicit-check-not=global_module
//
// RUN: %clang_cc1 -std=c++20 %t/user.cpp -triple %itanium_abi_triple -fmodule-file=%t/Module.pcm -emit-llvm -o - | FileCheck %t/user.cpp --implicit-check-not=unused --implicit-check-not=global_module

//--- Module.cppm
// CHECK-DAG: @extern_var_global_module = external {{(dso_local )?}}global
// CHECK-DAG: @inline_var_global_module = linkonce_odr {{(dso_local )?}}global
// CHECK-DAG: @_ZL24static_var_global_module = internal global
// CHECK-DAG: @_ZL23const_var_global_module = internal constant
//
// With strong-ownership, the module is mangled into exported symbols
// that are attached to a named module.
// CHECK-DAG: @_ZW6Module19extern_var_exported = external {{(dso_local )?}}global
// FIXME: Should this be 'weak_odr global'? Presumably it must be, since we
// can discard this global and its initializer (if any), and other TUs are not
// permitted to run the initializer for this variable.
// CHECK-DAG: @_ZW6Module19inline_var_exported = linkonce_odr {{(dso_local )?}}global
// CHECK-DAG: @_ZW6Module18const_var_exported = {{(dso_local )?}}constant
//
// CHECK-DAG: @_ZW6Module25extern_var_module_linkage = external {{(dso_local )?}}global
// FIXME: Should this be 'weak_odr global'? Presumably it must be, since we
// can discard this global and its initializer (if any), and other TUs are not
// permitted to run the initializer for this variable.
// CHECK-DAG: @_ZW6Module25inline_var_module_linkage = linkonce_odr {{(dso_local )?}}global
// CHECK-DAG: @_ZL25static_var_module_linkage = internal
// CHECK-DAG: @_ZL24const_var_module_linkage = internal
//
// CHECK-DAG: @_ZW6Module25unused_var_module_linkage = {{(dso_local )?}}global i32 4

module;

static void unused_static_global_module() {}
static void used_static_global_module() {}

inline void unused_inline_global_module() {}
inline void used_inline_global_module() {}

extern int extern_var_global_module;
inline int inline_var_global_module;
static int static_var_global_module;
const int const_var_global_module = 3;

// CHECK: define {{(dso_local )?}}void {{.*}}@_Z23noninline_global_modulev
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
  inline void unused_inline_exported() {}
  inline void used_inline_exported() {}

  extern int extern_var_exported;
  inline int inline_var_exported;
  const int const_var_exported = 3;

  // CHECK: define {{(dso_local )?}}void {{.*}}@_ZW6Module18noninline_exportedv
  void noninline_exported() {
    (void)&extern_var_exported;
    (void)&inline_var_exported;
    (void)&const_var_exported;
  }
}

static void unused_static_module_linkage() {}

static void used_static_module_linkage() {}

inline void unused_inline_module_linkage() {}
inline void used_inline_module_linkage() {}

extern int extern_var_module_linkage;
inline int inline_var_module_linkage;
static int static_var_module_linkage;
const int const_var_module_linkage = 3;

// CHECK: define {{(dso_local )?}}void {{.*}}@_ZW6Module24noninline_module_linkagev
// CHECK: define {{.*}}void {{.*}}@_ZL26used_static_module_linkagev
void noninline_module_linkage() {
  used_static_module_linkage();
  // CHECK: define linkonce_odr {{.*}}@_ZW6Module26used_inline_module_linkagev
  used_inline_module_linkage();

  (void)&extern_var_module_linkage;
  (void)&inline_var_module_linkage;
  (void)&static_var_module_linkage;
  (void)&const_var_module_linkage;
}

int unused_var_module_linkage = 4;
static int unused_static_var_module_linkage = 5;
inline int unused_inline_var_module_linkage = 6;
const int unused_const_var_module_linkage = 7;

struct a {
  struct b {};
  struct c {};
};
// CHECK: define {{(dso_local )?}}void @_ZW6Module1fNS_1a1bENS0_1cE(
void f(a::b, a::c) {}

//--- module.cpp

// CHECK-DAG: @_ZW6Module19extern_var_exported = external {{(dso_local )?}}global
// CHECK-DAG: @_ZW6Module19inline_var_exported = linkonce_odr {{(dso_local )?}}global
// CHECK-DAG: @_ZW6Module18const_var_exported = available_externally {{(dso_local )?}}constant i32 3,
//
// CHECK-DAG: @_ZW6Module25extern_var_module_linkage = external {{(dso_local )?}}global
// CHECK-DAG: @_ZW6Module25inline_var_module_linkage = linkonce_odr {{(dso_local )?}}global
// CHECK-DAG: @_ZL25static_var_module_linkage = internal {{(dso_local )?}}global i32 0,
// CHECK-DAG: @_ZL24const_var_module_linkage = internal {{(dso_local )?}}constant i32 3,

module Module;

void use() {
  // CHECK: define linkonce_odr {{.*}}@_ZW6Module20used_inline_exportedv
  used_inline_exported();
  // CHECK: declare {{.*}}@_ZW6Module18noninline_exportedv
  noninline_exported();

  (void)&extern_var_exported;
  (void)&inline_var_exported;
  (void)&const_var_exported;

  // CHECK: define {{.*}}@_ZL26used_static_module_linkagev
  used_static_module_linkage();

  // CHECK: define linkonce_odr {{.*}}@_ZW6Module26used_inline_module_linkagev
  used_inline_module_linkage();

  // CHECK: declare {{.*}}@_ZW6Module24noninline_module_linkagev
  noninline_module_linkage();

  (void)&extern_var_module_linkage;
  (void)&inline_var_module_linkage;
  (void)&static_var_module_linkage; // FIXME: Should not be visible here.
  (void)&const_var_module_linkage;
}

//--- user.cpp

// CHECK-DAG: @_ZW6Module19extern_var_exported = external {{(dso_local )?}}global
// CHECK-DAG: @_ZW6Module19inline_var_exported = linkonce_odr {{(dso_local )?}}global
// CHECK-DAG: @_ZW6Module18const_var_exported = available_externally {{(dso_local )?}}constant i32 3

import Module;

void use() {
  // CHECK: define linkonce_odr {{.*}}@_ZW6Module20used_inline_exportedv
  used_inline_exported();
  // CHECK: declare {{.*}}@_ZW6Module18noninline_exportedv
  noninline_exported();

  (void)&extern_var_exported;
  (void)&inline_var_exported;
  (void)&const_var_exported;

  // Module-linkage declarations are not visible here.
}
