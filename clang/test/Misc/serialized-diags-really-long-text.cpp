// Make sure that diagnostics serialization does not crash with a really long diagnostic text.

// RUN: not %clang_cc1 -std=c++11 %s -serialize-diagnostic-file %t.dia
// RUN: c-index-test -read-diagnostics %t.dia 2>&1 | FileCheck %s

typedef class AReallyLooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooongName {} alias;

template <int N, typename ...T>
struct MyTS {
  typedef MyTS<N-1, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias,
    alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias,
    alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias,
    alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias,
    alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias,
    alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias,
    alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias,
    alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias,
    alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, alias, T...> type;
  static type callme() {
    return type::callme();
  }
};

template <typename ...T>
struct MyTS<0, T...> {};

void foo() {
  // CHECK: [[@LINE+1]]:20: note: in instantiation of member function
  int e = MyTS<2>::callme();
}
