// RUN: %check_clang_tidy -std=c++17-or-later %s misc-unused-using-decls %t -- --fix-notes -- -fno-delayed-template-parsing

namespace ns {

template <typename T> class Foo {
public:
  Foo(T);
};
// Deduction guide (CTAD)
template <typename T> Foo(T t) -> Foo<T>;

template <typename T> class Bar {
public:
  Bar(T);
};

template <typename T> class Unused {};

} // namespace ns

using ns::Bar;
using ns::Foo;
using ns::Unused; // Unused
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: using decl 'Unused' is unused
// CHECK-FIXES: {{^}}// Unused

void f() {
  Foo(123);
  Bar(1);
}
