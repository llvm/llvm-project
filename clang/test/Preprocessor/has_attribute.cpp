// RUN: %clang_cc1 -triple i386-unknown-unknown -fms-compatibility -std=c++11 -E %s -o - | FileCheck %s

// CHECK: has_cxx11_carries_dep
#if __has_cpp_attribute(carries_dependency)
  int has_cxx11_carries_dep();
#endif

// CHECK: has_clang_fallthrough_1
#if __has_cpp_attribute(clang::fallthrough)
  int has_clang_fallthrough_1();
#endif

// CHECK: does_not_have_selectany
#if !__has_cpp_attribute(selectany)
  int does_not_have_selectany();
#endif

// The attribute name can be bracketed with double underscores.
// CHECK: has_clang_fallthrough_2
#if __has_cpp_attribute(clang::__fallthrough__)
  int has_clang_fallthrough_2();
#endif

// The scope cannot be bracketed with double underscores unless it is
// for gnu or clang.
// CHECK: does_not_have___gsl___suppress
#if !__has_cpp_attribute(__gsl__::suppress)
  int does_not_have___gsl___suppress();
#endif

// We do somewhat support the __clang__ vendor namespace, but it is a
// predefined macro and thus we encourage users to use _Clang instead.
// Because of this, we do not support __has_cpp_attribute for that
// vendor namespace.
// CHECK: does_not_have___clang___fallthrough
#if !__has_cpp_attribute(__clang__::fallthrough)
  int does_not_have___clang___fallthrough();
#endif

// CHECK: does_have_Clang_fallthrough
#if __has_cpp_attribute(_Clang::fallthrough)
  int does_have_Clang_fallthrough();
#endif

// CHECK: has_gnu_const
#if __has_cpp_attribute(__gnu__::__const__)
  int has_gnu_const();
#endif

// Test that C++11, target-specific attributes behave properly.

// CHECK: does_not_have_mips16
#if !__has_cpp_attribute(gnu::mips16)
  int does_not_have_mips16();
#endif

// Test that the version numbers of attributes listed in SD-6 are supported
// correctly.

// CHECK: has_cxx11_carries_dep_vers
#if __has_cpp_attribute(carries_dependency) == 200809
  int has_cxx11_carries_dep_vers();
#endif

// CHECK: has_cxx11_noreturn_vers
#if __has_cpp_attribute(noreturn) == 200809
  int has_cxx11_noreturn_vers();
#endif

// CHECK: has_cxx14_deprecated_vers
#if __has_cpp_attribute(deprecated) == 201309
  int has_cxx14_deprecated_vers();
#endif

// CHECK: has_cxx1z_nodiscard
#if __has_cpp_attribute(nodiscard) == 201603
  int has_cxx1z_nodiscard();
#endif

// CHECK: has_cxx1z_fallthrough
#if __has_cpp_attribute(fallthrough) == 201603
  int has_cxx1z_fallthrough();
#endif

// CHECK: has_declspec_uuid
#if __has_declspec_attribute(uuid)
  int has_declspec_uuid();
#endif

// CHECK: has_declspec_uuid2
#if __has_declspec_attribute(__uuid__)
  int has_declspec_uuid2();
#endif

// CHECK: does_not_have_declspec_fallthrough
#if !__has_declspec_attribute(fallthrough)
  int does_not_have_declspec_fallthrough();
#endif
