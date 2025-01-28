// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=cxx20 %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify=cxx23-26 %s
// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify=cxx23-26 %s

// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=cxx20    -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify=cxx23-26 -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify=cxx23-26 -fexperimental-new-constant-interpreter %s

// cxx23-26-no-diagnostics

namespace GH73232 {
namespace ex1 {
template <typename T>
constexpr void g(T); // #ex1-g-decl

constexpr int f() {
  g(0); // #ex1-g-call
  return 0;
}

template <typename T> 
constexpr void g(T) {}

constexpr auto z = f(); // #ex1-z-defn
// cxx20-error@-1 {{constexpr variable 'z' must be initialized by a constant expression}}
//   cxx20-note@#ex1-g-call {{undefined function 'g<int>' cannot be used in a constant expression}}
//   cxx20-note@#ex1-z-defn {{in call to 'f()'}}
//   cxx20-note@#ex1-g-decl {{declared here}}
} // namespace ex1

namespace ex2 {
template <typename> constexpr static void fromType();

void registerConverter() { fromType<int>(); }
template <typename> struct QMetaTypeId  {};
template <typename T> constexpr void fromType() { 
  (void)QMetaTypeId<T>{};
} // #1
template <> struct QMetaTypeId<int> {}; // #20428 
} // namespace ex2
} // namespace GH73232
