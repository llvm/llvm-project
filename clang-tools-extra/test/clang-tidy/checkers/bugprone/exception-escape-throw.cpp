// RUN: %check_clang_tidy -std=c++11,c++14 %s bugprone-exception-escape %t -- -- -fexceptions

void throwing_throw_nothing() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throwing_throw_nothing' which should not throw exceptions
  throw 1;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: note: frame #0: unhandled exception may be thrown in function 'throwing_throw_nothing' here

void explicit_int_thrower() throw(int);

void implicit_int_thrower() {
  throw 5;
}

void indirect_implicit() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_implicit' which should not throw exceptions
  implicit_int_thrower();
}
// CHECK-MESSAGES: :[[@LINE-7]]:3: note: frame #0: unhandled exception may be thrown in function 'implicit_int_thrower' here
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #1: function 'indirect_implicit' calls function 'implicit_int_thrower' here

void indirect_explicit() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_explicit' which should not throw exceptions
  explicit_int_thrower();
}
// CHECK-MESSAGES: :[[@LINE-17]]:29: note: frame #0: unhandled exception may be thrown in function 'explicit_int_thrower' here
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #1: function 'indirect_explicit' calls function 'explicit_int_thrower' here

struct super_throws {
  super_throws() throw(int) { throw 42; }
};

struct sub_throws : super_throws {
  sub_throws() throw() : super_throws() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'sub_throws' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-7]]:31: note: frame #0: unhandled exception may be thrown in function 'super_throws' here
// CHECK-MESSAGES: :[[@LINE-4]]:26: note: frame #1: function 'sub_throws' calls function 'super_throws' here
