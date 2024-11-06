// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s

namespace GH73232 {
namespace ex1 {
template <typename T>
constexpr void g(T);

constexpr int f() {
    g(0);
    return 0;
}

template <typename T> 
constexpr void g(T) {}

constexpr auto z = f();
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