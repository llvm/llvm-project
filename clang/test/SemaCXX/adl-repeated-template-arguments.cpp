// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace assoc {
struct Leaf {};
template<class T, class U=T> struct Node {};
void consume(...);
}
void test(assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Node<assoc::Leaf>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> value) { consume(value); }

template<class T> struct Base {};
namespace other { struct Tag {}; }
struct Derived : Base<other::Tag> {};
namespace other {
constexpr int select(Derived, Base<Tag>) { return 7; }
}
static_assert(select(Derived{}, Base<other::Tag>{}) == 7);
