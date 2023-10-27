// RUN: rm -rf %t
// RUN: %clang_cc1 -cxx-isystem %S/Inputs/builtin-headers/c++ -internal-isystem %S/Inputs/builtin-headers -fsyntax-only -fmodules -fmodules-cache-path=%t -fmodule-map-file=%S/Inputs/builtin-headers/c++/module.modulemap -fmodule-map-file=%resource_dir/module.modulemap -fmodule-map-file=%S/Inputs/builtin-headers/system-modules.modulemap -fbuiltin-headers-in-system-modules -DSYSTEM_MODULES %s -verify
// RUN: rm -rf %t
// RUN: %clang_cc1 -cxx-isystem %S/Inputs/builtin-headers/c++ -internal-isystem %S/Inputs/builtin-headers -fsyntax-only -fmodules -fmodules-cache-path=%t -fmodule-map-file=%S/Inputs/builtin-headers/c++/module.modulemap -fmodule-map-file=%resource_dir/module.modulemap -fmodule-map-file=%S/Inputs/builtin-headers/builtin-modules.modulemap %s -verify

// expected-no-diagnostics

@import cpp_stdint;

// The builtin modules are always available, though they're mostly
// empty if -fbuiltin-headers-in-system-modules is used.
@import _Builtin_float;
@import _Builtin_inttypes;
@import _Builtin_iso646;
@import _Builtin_limits;
@import _Builtin_stdalign;
@import _Builtin_stdarg;
@import _Builtin_stdatomic;
@import _Builtin_stdbool;
@import _Builtin_stddef;
@import _Builtin_stdint;
@import _Builtin_stdnoreturn;
@import _Builtin_tgmath;
@import _Builtin_unwind;

#ifdef SYSTEM_MODULES
// system-modules.modulemap uses the "mega module" style with
// -fbuiltin-headers-in-system-modules, and its modules cover
// the clang builtin headers.
@import cstd;
#else
// builtin-modules.modulemap uses top level modules for each
// of its headers, which allows interleaving with the builtin
// modules and libc++ modules.
@import c_complex;
@import c_float;
@import c_inttypes;
@import c_limits;
@import c_math;
@import c_stdint;
#endif
