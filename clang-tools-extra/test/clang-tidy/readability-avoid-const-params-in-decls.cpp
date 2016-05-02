// RUN: %check_clang_tidy %s readability-avoid-const-params-in-decls %t

using alias_type = bool;
using alias_const_type = const bool;


void F1(const int i);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: parameter 'i' is const-qualified in the function declaration; const-qualification of parameters only has an effect in function definitions [readability-avoid-const-params-in-decls]
// CHECK-FIXES: void F1(int i);

void F2(const int *const i);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: parameter 'i' is const-qualified
// CHECK-FIXES: void F2(const int *i);

void F3(int const i);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: parameter 'i' is const-qualified
// CHECK-FIXES: void F3(int i);

void F4(alias_type const i);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: parameter 'i' is const-qualified
// CHECK-FIXES: void F4(alias_type i);

void F5(const int);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: parameter 1 is const-qualified
// CHECK-FIXES: void F5(int);

void F6(const int *const);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: parameter 1 is const-qualified
// CHECK-FIXES: void F6(const int *);

void F7(int, const int);
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: parameter 2 is const-qualified
// CHECK-FIXES: void F7(int, int);

void F8(const int, const int);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: parameter 1 is const-qualified
// CHECK-MESSAGES: :[[@LINE-2]]:20: warning: parameter 2 is const-qualified
// CHECK-FIXES: void F8(int, int);

void F9(const int long_name);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: parameter 'long_name'
// CHECK-FIXES: void F9(int long_name);

void F10(const int *const *const long_name);
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: parameter 'long_name'
// CHECK-FIXES: void F10(const int *const *long_name);

void F11(const unsigned int /*v*/);
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: parameter 1
// CHECK-FIXES: void F11(unsigned int /*v*/);

void F12(const bool b = true);
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: parameter 'b'
// CHECK-FIXES: void F12(bool b = true);

struct Foo {
  Foo(const int i);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: parameter 'i'
  // CHECK-FIXES: Foo(int i);

  void operator()(const int i);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: parameter 'i'
  // CHECK-FIXES: void operator()(int i);
};

// Do not match on definitions
void NF1(const int i) {}
void NF2(const int *const i) {}
void NF3(int const i) {}
void NF4(alias_type const i) {}
void NF5(const int) {}
void NF6(const int *const) {}
void NF7(int, const int) {}
void NF8(const int, const int) {}

// Do not match on other stuff
void NF(const alias_type& i);
void NF(const int &i);
void NF(const int *i);
void NF(alias_const_type i);
void NF(const alias_type&);
void NF(const int&);
void NF(const int*);
void NF(const int* const*);
void NF(alias_const_type);
