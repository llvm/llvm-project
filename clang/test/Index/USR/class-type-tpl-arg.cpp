// RUN: c-index-test core -print-source-symbols -- -std=c++20 %s | FileCheck %s

// Check USRs of template specializations with class-type NTTP values.

struct R {
  int a;
};

template <R> struct L {};

// Equal parameter object values must produce equal USRs, and distinct values
// must produce distinct ones.

// CHECK: | f | c:@F@f#$@S@L>#@TPO@1$@S@R[[#HASH0:]]#
void f(L<R{0}>);

// CHECK: | g | c:@F@g#$@S@L>#@TPO@1$@S@R[[#HASH0]]#
void g(L<R{0}>);

// CHECK: | h | c:@F@h#$@S@L>#@TPO@1$@S@R[[#HASH1:]]#
// CHECK-NOT: | h | c:@F@h#$@S@L>#@TPO@1$@S@R[[#HASH0]]#
void h(L<R{1}>);

struct S {
  // CHECK: | set | c:@S@S@F@set#&&$@S@L>#@TPO@1$@S@R[[#HASH0]]#
  void set(L<R{0}> &&);
};
