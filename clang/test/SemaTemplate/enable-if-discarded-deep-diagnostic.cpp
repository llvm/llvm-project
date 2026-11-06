// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace std {
template<bool, class T=void> struct enable_if {};
template<class T> struct enable_if<true,T> { using type=T; };
template<bool B, class T=void> using enable_if_t=typename enable_if<B,T>::type;
}
template<class T, class U=T> struct Node {};
template<class T, std::enable_if_t<(sizeof(T) == 0), int> = 0> void accept(T) {}
void accept(...) {}
int main() { ::accept(Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<Node<int>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{}); }
