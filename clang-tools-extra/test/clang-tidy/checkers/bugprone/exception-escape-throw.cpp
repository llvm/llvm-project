// RUN: %check_clang_tidy -std=c++11,c++14 %s bugprone-exception-escape %t -- -- -fexceptions

void throwing_throw_nothing() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throwing_throw_nothing' which should not throw exceptions
  throw 1;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'throwing_throw_nothing' here

void explicit_int_thrower() throw(int);

void implicit_int_thrower() {
  throw 5;
}

void indirect_implicit() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_implicit' which should not throw exceptions
  implicit_int_thrower();
}
// CHECK-MESSAGES: :[[@LINE-7]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'implicit_int_thrower' here
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #1: function 'indirect_implicit' calls function 'implicit_int_thrower' here

void indirect_explicit() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_explicit' which should not throw exceptions
  explicit_int_thrower();
}
// CHECK-MESSAGES: :[[@LINE-17]]:29: note: frame #0: unhandled exception of type 'int' may be thrown in function 'explicit_int_thrower' here
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #1: function 'indirect_explicit' calls function 'explicit_int_thrower' here

struct super_throws {
  super_throws() throw(int) { throw 42; }
};

struct sub_throws : super_throws {
  sub_throws() throw() : super_throws() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'sub_throws' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-7]]:31: note: frame #0: unhandled exception of type 'int' may be thrown in function 'super_throws' here
// CHECK-MESSAGES: :[[@LINE-4]]:26: note: frame #1: function 'sub_throws' calls function 'super_throws' here

struct base_throwing_ctor {
  base_throwing_ctor() throw(int) { throw 123; }
};

struct intermediate_ctor : base_throwing_ctor {
  intermediate_ctor() throw(int) : base_throwing_ctor() {}
};

struct final_no_throw : intermediate_ctor {
  final_no_throw() throw() : intermediate_ctor() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'final_no_throw' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-11]]:37: note: frame #0: unhandled exception of type 'int' may be thrown in function 'base_throwing_ctor' here
// CHECK-MESSAGES: :[[@LINE-8]]:36: note: frame #1: function 'intermediate_ctor' calls function 'base_throwing_ctor' here
// CHECK-MESSAGES: :[[@LINE-5]]:30: note: frame #2: function 'final_no_throw' calls function 'intermediate_ctor' here

// Member initializer with call stack
struct member_thrower {
  member_thrower() throw(double) { throw 3.14; }
};

struct has_throwing_member {
  member_thrower member;
  has_throwing_member() throw() : member() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'has_throwing_member' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-8]]:36: note: frame #0: unhandled exception of type 'double' may be thrown in function 'member_thrower' here
// CHECK-MESSAGES: :[[@LINE-4]]:35: note: frame #1: function 'has_throwing_member' calls function 'member_thrower' here

void multi_spec_thrower() throw(int, double, const char*);

void calls_multi_spec() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_multi_spec' which should not throw exceptions
  multi_spec_thrower();
}
// CHECK-MESSAGES: :[[@LINE-6]]:27: note: frame #0: unhandled exception of type '{{(int|double|const char \*)}}' may be thrown in function 'multi_spec_thrower' here
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #1: function 'calls_multi_spec' calls function 'multi_spec_thrower' here
