// RUN: %clang_cc1 -std=c++14 %s -verify -fms-extensions -triple x86_64-windows-msvc
// RUN: %clang_cc1 -std=c++17 %s -verify -fms-extensions -triple x86_64-windows-msvc

__declspec(dllimport) void imported_func();
__declspec(dllimport) int imported_int;
struct Foo {
  void __declspec(dllimport) imported_method();
};

// Instantiation is OK.
template <void (*FP)()> struct TemplateFnPtr {
  static void getit() { FP(); }
};
template <void (&FP)()> struct TemplateFnRef {
  static void getit() { FP(); }
};
void instantiate1() {
  TemplateFnPtr<&imported_func>::getit();
  TemplateFnRef<imported_func>::getit();
}

// Check variable template instantiation.
template <int *GI> struct TemplateIntPtr {
  static int getit() { return *GI; }
};
template <int &GI> struct TemplateIntRef {
  static int getit() { return GI; }
};
int instantiate2() {
  int r = 0;
  r += TemplateIntPtr<&imported_int>::getit();
  r += TemplateIntRef<imported_int>::getit();
  return r;
}

// Member pointer instantiation.
template <void (Foo::*MP)()> struct TemplateMemPtr { };
TemplateMemPtr<&Foo::imported_method> instantiate_mp;

// constexpr initialization doesn't work for dllimport things.
// expected-error@+1{{must be initialized by a constant expression}}
constexpr void (*constexpr_import_func)() = &imported_func;
// expected-error@+1{{must be initialized by a constant expression}}
constexpr int *constexpr_import_int = &imported_int;
// expected-error@+1{{must be initialized by a constant expression}}
constexpr void (Foo::*constexpr_memptr)() = &Foo::imported_method;

// We make dynamic initializers for 'const' globals, but not constexpr ones.
void (*const const_import_func)() = &imported_func;
int *const const_import_int = &imported_int;
void (Foo::*const const_memptr)() = &Foo::imported_method;

// Check that using a non-type template parameter for constexpr global
// initialization is correctly diagnosed during template instantiation.
template <void (*FP)()> struct StaticConstexpr {
  // expected-error@+1{{must be initialized by a constant expression}}
  static constexpr void (*g_fp)() = FP;
};
void instantiate3() {
  // expected-note@+1 {{requested here}}
  StaticConstexpr<imported_func>::g_fp();
}
