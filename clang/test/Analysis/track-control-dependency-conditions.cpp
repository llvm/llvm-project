// RUN: %clang_analyze_cc1 %s \
// RUN:   -verify=expected,tracking \
// RUN:   -analyzer-config track-conditions=true \
// RUN:   -analyzer-output=text \
// RUN:   -analyzer-checker=core

// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config track-conditions-debug=true \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-DEBUG

// CHECK-INVALID-DEBUG: (frontend): invalid input for analyzer-config option
// CHECK-INVALID-DEBUG-SAME:        'track-conditions-debug', that expects
// CHECK-INVALID-DEBUG-SAME:        'track-conditions' to also be enabled
//
// RUN: %clang_analyze_cc1 %s \
// RUN:   -verify=expected,tracking,debug \
// RUN:   -analyzer-config track-conditions=true \
// RUN:   -analyzer-config track-conditions-debug=true \
// RUN:   -analyzer-output=text \
// RUN:   -analyzer-checker=core

// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-output=text \
// RUN:   -analyzer-checker=core

namespace example_1 {
int flag;
bool coin();

void foo() {
  flag = coin(); // tracking-note{{Value assigned to 'flag'}}
}

void test() {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}
  flag = 1;

  foo(); // TODO: Add nodes here about flag's value being invalidated.
  if (flag) // expected-note   {{Assuming 'flag' is 0}}
            // expected-note@-1{{Taking false branch}}
    x = new int;

  foo(); // tracking-note{{Calling 'foo'}}
         // tracking-note@-1{{Returning from 'foo'}}

  if (flag) // expected-note   {{Assuming 'flag' is not equal to 0}}
            // expected-note@-1{{Taking true branch}}
            // debug-note@-2{{Tracking condition 'flag'}}

    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace example_1

namespace example_2 {
int flag;
bool coin();

void foo() {
  flag = coin(); // tracking-note{{Value assigned to 'flag'}}
}

void test() {
  int *x = 0;
  flag = 1;

  foo();
  if (flag) // expected-note   {{Assuming 'flag' is 0}}
            // expected-note@-1{{Taking false branch}}
    x = new int;

  x = 0; // expected-note{{Null pointer value stored to 'x'}}

  foo(); // tracking-note{{Calling 'foo'}}
         // tracking-note@-1{{Returning from 'foo'}}

  if (flag) // expected-note   {{Assuming 'flag' is not equal to 0}}
            // expected-note@-1{{Taking true branch}}
            // debug-note@-2{{Tracking condition 'flag'}}

    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace example_2

namespace global_variable_invalidation {
int flag;
bool coin();

void foo() {
  // coin() could write bar, do it's invalidated.
  flag = coin(); // tracking-note{{Value assigned to 'flag'}}
                 // tracking-note@-1{{Value assigned to 'bar'}}
}

int bar;

void test() {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}
  flag = 1;

  foo(); // tracking-note{{Calling 'foo'}}
         // tracking-note@-1{{Returning from 'foo'}}

  if (bar) // expected-note   {{Assuming 'bar' is not equal to 0}}
           // expected-note@-1{{Taking true branch}}
           // debug-note@-2{{Tracking condition 'bar'}}
    if (flag) // expected-note   {{Assuming 'flag' is not equal to 0}}
              // expected-note@-1{{Taking true branch}}
              // debug-note@-2{{Tracking condition 'flag'}}

      *x = 5; // expected-warning{{Dereference of null pointer}}
              // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace global_variable_invalidation

namespace variable_declaration_in_condition {
bool coin();

bool foo() {
  return coin(); // tracking-note{{Returning value}}
}

int bar;

void test() {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}

  if (int flag = foo()) // tracking-note{{Calling 'foo'}}
                        // tracking-note@-1{{Returning from 'foo'}}
                        // tracking-note@-2{{'flag' initialized here}}
                        // debug-note@-3{{Tracking condition 'flag'}}
                        // expected-note@-4{{Assuming 'flag' is not equal to 0}}
                        // expected-note@-5{{Taking true branch}}

    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace variable_declaration_in_condition

namespace conversion_to_bool {
bool coin();

struct ConvertsToBool {
  operator bool() const { return coin(); } // tracking-note{{Returning value}}
};

void test() {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}

  if (ConvertsToBool())
    // tracking-note@-1 {{Calling 'ConvertsToBool::operator bool'}}
    // tracking-note@-2{{Returning from 'ConvertsToBool::operator bool'}}
    // debug-note@-3{{Tracking condition 'ConvertsToBool()'}}
    // expected-note@-4{{Assuming the condition is true}}
    // expected-note@-5{{Taking true branch}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace variable_declaration_in_condition

namespace unimportant_returning_value_note {
bool coin();

bool flipCoin() { return coin(); } // tracking-note{{Returning value}}

void i(int *ptr) {
  if (ptr) // expected-note{{Assuming 'ptr' is null}}
           // expected-note@-1{{Taking false branch}}
    ;
  if (!flipCoin())
    // tracking-note@-1{{Calling 'flipCoin'}}
    // tracking-note@-2{{Returning from 'flipCoin'}}
    // debug-note@-3{{Tracking condition '!flipCoin()'}}
    // expected-note@-4{{Assuming the condition is true}}
    // expected-note@-5{{Taking true branch}}
    *ptr = 5; // expected-warning{{Dereference of null pointer}}
              // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace unimportant_returning_value_note

namespace important_returning_value_note {
bool coin();

bool flipCoin() {
  if (coin()) // tracking-note{{Assuming the condition is false}}
              // tracking-note@-1{{Taking false branch}}
              // debug-note@-2{{Tracking condition 'coin()'}}
    return true;
  return coin(); // tracking-note{{Returning value}}
}

void i(int *ptr) {
  if (ptr) // expected-note{{Assuming 'ptr' is null}}
           // expected-note@-1{{Taking false branch}}
    ;
  if (!flipCoin())
    // tracking-note@-1{{Calling 'flipCoin'}}
    // tracking-note@-2{{Returning from 'flipCoin'}}
    // debug-note@-3{{Tracking condition '!flipCoin()'}}
    // expected-note@-4{{Assuming the condition is true}}
    // expected-note@-5{{Taking true branch}}
    *ptr = 5; // expected-warning{{Dereference of null pointer}}
              // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace important_returning_value_note

namespace tracked_condition_is_only_initialized {
int getInt();

void f() {
  int flag = getInt(); // tracking-note{{'flag' initialized here}}
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}
  if (flag) // expected-note{{Assuming 'flag' is not equal to 0}}
            // expected-note@-1{{Taking true branch}}
            // debug-note@-2{{Tracking condition 'flag'}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace tracked_condition_is_only_initialized

namespace tracked_condition_written_in_same_stackframe {
int flag;
int getInt();

void f(int y) {
  y = 1; // tracking-note{{The value 1 is assigned to 'y'}}
  flag = y; // tracking-note{{The value 1 is assigned to 'flag'}}

  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}
  if (flag) // expected-note{{'flag' is 1}}
            // expected-note@-1{{Taking true branch}}
            // debug-note@-2{{Tracking condition 'flag'}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace tracked_condition_written_in_same_stackframe

namespace tracked_condition_written_in_nested_stackframe {
int flag;
int getInt();

void foo() {
  int y;
  y = 1; // tracking-note{{The value 1 is assigned to 'y'}}
  flag = y; // tracking-note{{The value 1 is assigned to 'flag'}}

}

void f(int y) {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}

  foo(); // tracking-note{{Calling 'foo'}}
         // tracking-note@-1{{Returning from 'foo'}}

  if (flag) // expected-note{{'flag' is 1}}
            // expected-note@-1{{Taking true branch}}
            // debug-note@-2{{Tracking condition 'flag'}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace tracked_condition_written_in_nested_stackframe

namespace collapse_point_not_in_condition {

[[noreturn]] void halt();

void assert(int b) {
  if (!b) // tracking-note{{Assuming 'b' is not equal to 0}}
          // tracking-note@-1{{Taking false branch}}
    halt();
}

void f(int flag) {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}

  assert(flag); // tracking-note{{Calling 'assert'}}
                // tracking-note@-1{{Returning from 'assert'}}

  if (flag) // expected-note{{'flag' is not equal to 0}}
            // expected-note@-1{{Taking true branch}}
            // debug-note@-2{{Tracking condition 'flag'}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace collapse_point_not_in_condition

namespace unimportant_write_before_collapse_point {

[[noreturn]] void halt();

void assert(int b) {
  if (!b) // tracking-note{{Assuming 'b' is not equal to 0}}
          // tracking-note@-1{{Taking false branch}}
    halt();
}
int getInt();

void f(int flag) {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}

  flag = getInt(); // tracking-note{{Value assigned to 'flag'}}
  assert(flag); // tracking-note{{Calling 'assert'}}
                // tracking-note@-1{{Returning from 'assert'}}

  if (flag) // expected-note{{'flag' is not equal to 0}}
            // expected-note@-1{{Taking true branch}}
            // debug-note@-2{{Tracking condition 'flag'}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace unimportant_write_before_collapse_point
