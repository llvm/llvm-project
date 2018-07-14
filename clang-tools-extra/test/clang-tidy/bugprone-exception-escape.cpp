// RUN: %check_clang_tidy %s bugprone-exception-escape %t -- -extra-arg=-std=c++11 -extra-arg=-fexceptions -config="{CheckOptions: [{key: bugprone-exception-escape.IgnoredExceptions, value: 'ignored1,ignored2'}, {key: bugprone-exception-escape.FunctionsThatShouldNotThrow, value: 'enabled1,enabled2,enabled3'}]}" --

struct throwing_destructor {
  ~throwing_destructor() {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function '~throwing_destructor' which should not throw exceptions
    throw 1;
  }
};

struct throwing_move_constructor {
  throwing_move_constructor(throwing_move_constructor&&) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'throwing_move_constructor' which should not throw exceptions
    throw 1;
  }
};

struct throwing_move_assignment {
  throwing_move_assignment& operator=(throwing_move_assignment&&) {
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: an exception may be thrown in function 'operator=' which should not throw exceptions
    throw 1;
  }
};

void throwing_noexcept() noexcept {
    // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throwing_noexcept' which should not throw exceptions
  throw 1;
}

void throwing_throw_nothing() throw() {
    // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throwing_throw_nothing' which should not throw exceptions
  throw 1;
}

void throw_and_catch() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_and_catch' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
  }
}

void throw_and_catch_some(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_and_catch_some' which should not throw exceptions
  try {
    if (n) throw 1;
    throw 1.1;
  } catch(int &) {
  }
}

void throw_and_catch_each(int n) noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_and_catch_each' which should not throw exceptions
  try {
    if (n) throw 1;
    throw 1.1;
  } catch(int &) {
  } catch(double &) {
  }
}

void throw_and_catch_all(int n) noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_and_catch_all' which should not throw exceptions
  try {
    if (n) throw 1;
    throw 1.1;
  } catch(...) {
  }
}

void throw_and_rethrow() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_and_rethrow' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
    throw;
  }
}

void throw_catch_throw() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_catch_throw' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
    throw 2;
  }
}

void throw_catch_rethrow_the_rest(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_catch_rethrow_the_rest' which should not throw exceptions
  try {
    if (n) throw 1;
    throw 1.1;
  } catch(int &) {
  } catch(...) {
    throw;
  }
}

class base {};
class derived: public base {};

void throw_derived_catch_base() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base' which should not throw exceptions
  try {
    throw derived();
  } catch(base &) {
  }
}

void try_nested_try(int n) noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'try_nested_try' which should not throw exceptions
  try {
    try {
      if (n) throw 1;
      throw 1.1;
    } catch(int &) {
    }
  } catch(double &) {
  }
}

void bad_try_nested_try(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'bad_try_nested_try' which should not throw exceptions
  try {
    if (n) throw 1;
    try {
      throw 1.1;
    } catch(int &) {
    }
  } catch(double &) {
  }
}

void try_nested_catch() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'try_nested_catch' which should not throw exceptions
  try {
    try {
      throw 1;
    } catch(int &) {
      throw 1.1;
    }
  } catch(double &) {
  }
}

void catch_nested_try() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'catch_nested_try' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
    try {
      throw 1;
    } catch(int &) {
    }
  }
}

void bad_catch_nested_try() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'bad_catch_nested_try' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
    try {
      throw 1.1;
    } catch(int &) {
    }
  } catch(double &) {
  }
}

void implicit_int_thrower() {
  throw 1;
}

void explicit_int_thrower() throw(int);

void indirect_implicit() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_implicit' which should not throw exceptions
  implicit_int_thrower();
}

void indirect_explicit() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_explicit' which should not throw exceptions
  explicit_int_thrower();
}

void indirect_catch() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_catch' which should not throw exceptions
  try {
    implicit_int_thrower();
  } catch(int&) {
  }
}

template<typename T>
void dependent_throw() noexcept(sizeof(T)<4) {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'dependent_throw' which should not throw exceptions
  if (sizeof(T) > 4)
    throw 1;
}

void swap(int&, int&) {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'swap' which should not throw exceptions
  throw 1;
}

namespace std {
class bad_alloc {};
}

void alloc() {
  throw std::bad_alloc();
}

void allocator() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'allocator' which should not throw exceptions
  alloc();
}

void enabled1() {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'enabled1' which should not throw exceptions
  throw 1;
}

void enabled2() {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'enabled2' which should not throw exceptions
  enabled1();
}

void enabled3() {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'enabled3' which should not throw exceptions
  try {
    enabled1();
  } catch(...) {
  }
}

class ignored1 {};
class ignored2 {};

void this_does_not_count() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'this_does_not_count' which should not throw exceptions
  throw ignored1();
}

void this_does_not_count_either(int n) noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'this_does_not_count_either' which should not throw exceptions
  try {
    throw 1;
    if (n) throw ignored2();
  } catch(int &) {
  }
}

void this_counts(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'this_counts' which should not throw exceptions
  if (n) throw 1;
  throw ignored1();
}

int main() {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'main' which should not throw exceptions
  throw 1;
  return 0;
}
