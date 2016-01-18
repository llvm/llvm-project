// RUN: not %clang_cc1 -fsyntax-only -fcolor-diagnostics %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -fcolor-diagnostics -fdiagnostics-show-template-tree %s 2>&1 | FileCheck %s -check-prefix=TREE
// REQUIRES: ansi-escape-sequences
template<typename> struct foo {};
void func(foo<int>);
int main() {
  func(foo<double>());
}
// CHECK: {{.*}}candidate function not viable: no known conversion from 'foo<[[CYAN:.\[0;1;36m]]double[[RESET:.\[0m]]>' to 'foo<[[CYAN]]int[[RESET]]>' for 1st argument[[RESET]]
// TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:  foo<
// TREE:    {{\[}}[[CYAN:.\[0;1;36m]]double[[RESET:.\[0m]] != [[CYAN]]int[[RESET]]]>[[RESET]]

foo<int> A;
foo<double> &B = A;
// CHECK: {{.*}}non-const lvalue reference to type 'foo<[[CYAN]]double[[RESET]][[BOLD:.\[1m]]>' cannot bind to a value of unrelated type 'foo<[[CYAN]]int[[RESET]][[BOLD]]>'[[RESET]]
// TREE: non-const lvalue reference cannot bind to a value of unrelated type
// TREE:   foo<
// TREE:     {{\[}}[[CYAN]]double[[RESET]][[BOLD:.\[1m]] != [[CYAN]]int[[RESET]][[BOLD]]]>[[RESET]]

template<typename> class vector {};

void set15(vector<const vector<int> >) {}
void test15() {
  set15(vector<const vector<const int> >());
}
// CHECK: {{.*}}candidate function not viable: no known conversion from 'vector<const vector<[[CYAN]]const{{ ?}}[[RESET]]{{ ?}}int>>' to 'vector<const vector<int>>' for 1st argument
// TREE: {{.*}}candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:   vector<
// TREE:     const vector<
// TREE:       {{\[}}[[CYAN]]const{{ ?}}[[RESET]]{{ ?}}!= [[CYAN]](no qualifiers)[[RESET]]] int>>

void set16(vector<vector<int> >) {}
void test16() {
  set16(vector<const vector<int> >());
}
// CHECK: {{.*}}candidate function not viable: no known conversion from 'vector<[[CYAN]]const{{ ?}}[[RESET]]{{ ?}}vector<[...]>>' to 'vector<vector<[...]>>' for 1st argument
// TREE: {{.*}}candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:   vector<
// TREE:     {{\[}}[[CYAN]]const{{ ?}}[[RESET]]{{ ?}}!= [[CYAN]](no qualifiers){{ ?}}[[RESET]]]{{ ?}}vector<
// TREE:       [...]>>

void set17(vector<const vector<int> >) {}
void test17() {
  set17(vector<vector<int> >());
}
// CHECK: candidate function not viable: no known conversion from 'vector<vector<[...]>>' to 'vector<[[CYAN]]const{{ ?}}[[RESET]]{{ ?}}vector<[...]>>' for 1st argument
// TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:   vector<
// TREE:     {{\[}}[[CYAN]](no qualifiers){{ ?}}[[RESET]]{{ ?}}!= [[CYAN]]const[[RESET]]] vector<
// TREE:       [...]>>

void set18(vector<volatile vector<int> >) {}
void test18() {
  set18(vector<const vector<int> >());
}
// CHECK: candidate function not viable: no known conversion from 'vector<[[CYAN]]const{{ ?}}[[RESET]]{{ ?}}vector<[...]>>' to 'vector<[[CYAN]]volatile{{ ?}}[[RESET]]{{ ?}}vector<[...]>>' for 1st argument
// TREE: no matching function for call to 'set18'
// TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:   vector<
// TREE:     {{\[}}[[CYAN]]const{{ ?}}[[RESET]]{{ ?}}!= [[CYAN]]volatile[[RESET]]] vector<
// TREE:       [...]>>

void set19(vector<const volatile vector<int> >) {}
void test19() {
  set19(vector<const vector<int> >());
}
// CHECK: candidate function not viable: no known conversion from 'vector<const vector<[...]>>' to 'vector<const [[CYAN]]volatile{{ ?}}[[RESET]]{{ ?}}vector<[...]>>' for 1st argument
// TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:   vector<
// TREE:     [const != const [[CYAN]]volatile[[RESET]]] vector<
// TREE:       [...]>>

namespace default_args {
  template <int x, int y = 1+1, int z = 2>
  class A {};

  void foo(A<0> &M) {
    // CHECK: no viable conversion from 'A<[...], (default) [[CYAN]]1 + 1[[RESET]][[BOLD]] aka [[CYAN]]2[[RESET]][[BOLD]], (default) [[CYAN]]2[[RESET]][[BOLD]]>' to 'A<[...], [[CYAN]]0[[RESET]][[BOLD]], [[CYAN]]0[[RESET]][[BOLD]]>'
    A<0, 0, 0> N = M;

    // CHECK: no viable conversion from 'A<[2 * ...], (default) [[CYAN]]2[[RESET]][[BOLD]]>' to 'A<[2 * ...], [[CYAN]]0[[RESET]][[BOLD]]>'
    A<0, 2, 0> N2 = M;
  }

}
