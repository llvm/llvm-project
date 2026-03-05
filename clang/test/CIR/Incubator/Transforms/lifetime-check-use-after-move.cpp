// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fclangir-lifetime-check="history=invalid,null" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

namespace std {
template <typename T>
T&& move(T& t) {
  return static_cast<T&&>(t);
}
}

void consume_int(int&&);
void consume_double(double&&);
void consume_float(float&&);

// Test 1: Basic int move
void test_int_basic() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int b = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 2: Multiple types
void test_double() {
  double d = 3.14;
  consume_double(std::move(d)); // expected-note {{moved here via std::move or rvalue reference}}
  double e = d; // expected-warning {{use of moved-from value 'd'}}
}

void test_float() {
  float f = 1.5f;
  consume_float(std::move(f)); // expected-note {{moved here via std::move or rvalue reference}}
  float g = f; // expected-warning {{use of moved-from value 'f'}}
}

// Test 4: Negative cases - NOT moves
void take_lvalue(int&);
void take_value(int);

void test_lvalue_ref() {
  int a = 10;
  take_lvalue(a); // Not a move
  int b = a; // OK
}

void test_by_value() {
  int a = 10;
  take_value(a); // Not a move (copies value)
  int b = a; // OK
}

// Test 5: Use in expressions
void test_use_in_expr() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int b = a + 5; // expected-warning {{use of moved-from value 'a'}}
}

int test_use_in_return() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  return a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 6: Multiple uses after move
void test_multiple_uses() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int b = a; // expected-warning {{use of moved-from value 'a'}}
  int c = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 7: Move in conditional
void test_move_in_if(bool cond) {
  int a = 10;
  if (cond) {
    consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  }
  int b = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 8: Move with different primitive types
void consume_char(char&&);
void consume_bool(bool&&);

void test_char() {
  char c = 'x';
  consume_char(std::move(c)); // expected-note {{moved here via std::move or rvalue reference}}
  char d = c; // expected-warning {{use of moved-from value 'c'}}
}

void test_bool() {
  bool b = true;
  consume_bool(std::move(b)); // expected-note {{moved here via std::move or rvalue reference}}
  bool c = b; // expected-warning {{use of moved-from value 'b'}}
}

// Test 8: Conditional move
void test_conditional_move(bool cond) {
  int a = 10;
  if (cond) {
    consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  }
  int b = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 10: Move-after-move
void test_move_after_move() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  consume_int(std::move(a)); // expected-warning {{use of moved-from value 'a'}}
}

// Test 16: Function parameter move
void test_parameter_move(int a) {
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int b = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 17: Loop with conditional move
void test_loop_with_move() {
  int a = 10;
  for (int i = 0; i < 3; i++) {
    if (i == 1) {
      consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
                                  // expected-warning@-1 {{use of moved-from value 'a'}}
    }
    if (i == 2) {
      int b = a; // expected-warning {{use of moved-from value 'a'}}
    }
  }
}

// Test 15: Switch with fallthrough
void test_switch_fallthrough(int cond) {
  int a = 10;
  switch (cond) {
  case 1:
    consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  case 2: // fallthrough
    int b = a; // expected-warning {{use of moved-from value 'a'}}
    break;
  }
}

// Test 21: Move in declaration
void test_move_in_declaration() {
  int a = 10;
  int b(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int c = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 22: Warn at every use location (consistent with invalid pointer behavior)
void test_warn_at_every_use() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int b = a; // expected-warning {{use of moved-from value 'a'}}
  int c = a; // expected-warning {{use of moved-from value 'a'}}
}
