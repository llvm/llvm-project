// RUN: %clang_cc1 -fsyntax-only -fcolor-diagnostics %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fcolor-diagnostics -fdiagnostics-show-template-tree %s 2>&1 | FileCheck %s -check-prefix=TREE
// REQUIRES: ansi-escape-sequences
template<typename> struct foo {};
void func(foo<int>);
int main() {
  func(foo<double>());
}
// CHECK: {{.*}}candidate function not viable: no known conversion from 'foo<{{.}}[0;1;36mdouble{{.}}[0m>' to 'foo<{{.}}[0;1;36mint{{.}}[0m>' for 1st argument{{.}}[0m
// TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:  foo<
// TREE:    [{{.}}[0;1;36mdouble{{.}}[0m != {{.}}[0;1;36mint{{.}}[0m]>{{.}}[0m

foo<int> A;
foo<double> &B = A;
// CHECK: {{.*}}non-const lvalue reference to type 'foo<{{.}}[0;1;36mdouble{{.}}[0m{{.}}[1m>' cannot bind to a value of unrelated type 'foo<{{.}}[0;1;36mint{{.}}[0m{{.}}[1m>'{{.}}[0m
// TREE: non-const lvalue reference cannot bind to a value of unrelated type
// TREE:   foo<
// TREE:     [{{.}}[0;1;36mdouble{{.}}[0m{{.}}[1m != {{.}}[0;1;36mint{{.}}[0m{{.}}[1m]>{{.}}[0m

template<typename> class vector {};

void set15(vector<const vector<int> >) {}
void test15() {
  set15(vector<const vector<const int> >());
}
// CHECK: {{.*}}candidate function not viable: no known conversion from 'vector<const vector<{{.}}[0;1;36mconst{{ ?.}}[0m{{ ?}}int>>' to 'vector<const vector<int>>' for 1st argument
// TREE: {{.*}}candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:   vector<
// TREE:     const vector<
// TREE:       [{{.}}[0;1;36mconst{{ ?.}}[0m{{ ?}}!= {{.}}[0;1;36m(no qualifiers){{.}}[0m] int>>

void set16(vector<vector<int> >) {}
void test16() {
  set16(vector<const vector<int> >());
}
// CHECK: {{.*}}candidate function not viable: no known conversion from 'vector<{{.}}[0;1;36mconst{{ ?.}}[0m{{ ?}}vector<[...]>>' to 'vector<vector<[...]>>' for 1st argument
// TREE: {{.*}}candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:   vector<
// TREE:     [{{.}}[0;1;36mconst{{ ?.}}[0m{{ ?}}!= {{.}}[0;1;36m(no qualifiers){{ ?.}}[0m]{{ ?}}vector<
// TREE:       [...]>>

void set17(vector<const vector<int> >) {}
void test17() {
  set17(vector<vector<int> >());
}
// CHECK: candidate function not viable: no known conversion from 'vector<vector<[...]>>' to 'vector<{{.}}[0;1;36mconst{{ ?.}}[0m{{ ?}}vector<[...]>>' for 1st argument
// TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:   vector<
// TREE:     [{{.}}[0;1;36m(no qualifiers){{ ?.}}[0m{{ ?}}!= {{.}}[0;1;36mconst{{.}}[0m] vector<
// TREE:       [...]>>

void set18(vector<volatile vector<int> >) {}
void test18() {
  set18(vector<const vector<int> >());
}
// CHECK: candidate function not viable: no known conversion from 'vector<{{.}}[0;1;36mconst{{ ?.}}[0m{{ ?}}vector<[...]>>' to 'vector<{{.}}[0;1;36mvolatile{{ ?.}}[0m{{ ?}}vector<[...]>>' for 1st argument
// TREE: no matching function for call to 'set18'
// TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:   vector<
// TREE:     [{{.}}[0;1;36mconst{{ ?.}}[0m{{ ?}}!= {{.}}[0;1;36mvolatile{{.}}[0m] vector<
// TREE:       [...]>>

void set19(vector<const volatile vector<int> >) {}
void test19() {
  set19(vector<const vector<int> >());
}
// CHECK: candidate function not viable: no known conversion from 'vector<const vector<[...]>>' to 'vector<const {{.}}[0;1;36mvolatile{{ ?.}}[0m{{ ?}}vector<[...]>>' for 1st argument
// TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:   vector<
// TREE:     [const != const {{.}}[0;1;36mvolatile{{.}}[0m] vector<
// TREE:       [...]>>
