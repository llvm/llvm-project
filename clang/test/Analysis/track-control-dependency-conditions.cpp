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
  return coin();
}

int bar;

void test() {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}

  if (int flag = foo()) // debug-note{{Tracking condition 'flag'}}
                        // expected-note@-1{{Assuming 'flag' is not equal to 0}}
                        // expected-note@-2{{Taking true branch}}

    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace variable_declaration_in_condition

namespace conversion_to_bool {
bool coin();

struct ConvertsToBool {
  operator bool() const { return coin(); }
};

void test() {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}

  if (ConvertsToBool())
    // debug-note@-1{{Tracking condition 'ConvertsToBool()'}}
    // expected-note@-2{{Assuming the condition is true}}
    // expected-note@-3{{Taking true branch}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace variable_declaration_in_condition

namespace note_from_different_but_not_nested_stackframe {

void nullptrDeref(int *ptr, bool True) {
  if (True) // expected-note{{'True' is true}}
            // expected-note@-1{{Taking true branch}}
            // debug-note@-2{{Tracking condition 'True}}
    *ptr = 5;
  // expected-note@-1{{Dereference of null pointer (loaded from variable 'ptr')}}
  // expected-warning@-2{{Dereference of null pointer (loaded from variable 'ptr')}}
}

void f() {
  int *ptr = nullptr;
  // expected-note@-1{{'ptr' initialized to a null pointer value}}
  bool True = true;
  nullptrDeref(ptr, True);
  // expected-note@-1{{Passing null pointer value via 1st parameter 'ptr'}}
  // expected-note@-2{{Calling 'nullptrDeref'}}
}

} // end of namespace note_from_different_but_not_nested_stackframe

namespace important_returning_pointer_loaded_from {
bool coin();

int *getIntPtr();

void storeValue(int **i) {
  *i = getIntPtr(); // tracking-note{{Value assigned to 'i'}}
}

int *conjurePointer() {
  int *i;
  storeValue(&i); // tracking-note{{Calling 'storeValue'}}
                  // tracking-note@-1{{Returning from 'storeValue'}}
  return i;       // tracking-note{{Returning pointer (loaded from 'i')}}
}

void f(int *ptr) {
  if (ptr) // expected-note{{Assuming 'ptr' is null}}
           // expected-note@-1{{Taking false branch}}
    ;
  if (!conjurePointer())
    // tracking-note@-1{{Calling 'conjurePointer'}}
    // tracking-note@-2{{Returning from 'conjurePointer'}}
    // debug-note@-3{{Tracking condition '!conjurePointer()'}}
    // expected-note@-4{{Assuming the condition is true}}
    // expected-note@-5{{Taking true branch}}
    *ptr = 5; // expected-warning{{Dereference of null pointer}}
              // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace important_returning_pointer_loaded_from

namespace unimportant_returning_pointer_loaded_from {
bool coin();

int *getIntPtr();

int *conjurePointer() {
  int *i = getIntPtr();
  return i;
}

void f(int *ptr) {
  if (ptr) // expected-note{{Assuming 'ptr' is null}}
           // expected-note@-1{{Taking false branch}}
    ;
  if (!conjurePointer())
    // debug-note@-1{{Tracking condition '!conjurePointer()'}}
    // expected-note@-2{{Assuming the condition is true}}
    // expected-note@-3{{Taking true branch}}
    *ptr = 5; // expected-warning{{Dereference of null pointer}}
              // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace unimportant_returning_pointer_loaded_from

namespace unimportant_returning_pointer_loaded_from_through_cast {

void *conjure();

int *cast(void *P) {
  return static_cast<int *>(P);
}

void f() {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}

  if (cast(conjure()))
    // debug-note@-1{{Tracking condition 'cast(conjure())'}}
    // expected-note@-2{{Assuming the condition is false}}
    // expected-note@-3{{Taking false branch}}
    return;
  *x = 5; // expected-warning{{Dereference of null pointer}}
          // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace unimportant_returning_pointer_loaded_from_through_cast

namespace unimportant_returning_value_note {
bool coin();

bool flipCoin() { return coin(); }

void i(int *ptr) {
  if (ptr) // expected-note{{Assuming 'ptr' is null}}
           // expected-note@-1{{Taking false branch}}
    ;
  if (!flipCoin())
    // debug-note@-1{{Tracking condition '!flipCoin()'}}
    // expected-note@-2{{Assuming the condition is true}}
    // expected-note@-3{{Taking true branch}}
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

namespace important_returning_value_note_in_linear_function {
bool coin();

struct super_complicated_template_hackery {
  static constexpr bool value = false;
};

bool flipCoin() {
  if (super_complicated_template_hackery::value)
    // tracking-note@-1{{'value' is false}}
    // tracking-note@-2{{Taking false branch}}
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
} // end of namespace important_returning_value_note_in_linear_function

namespace tracked_condition_is_only_initialized {
int getInt();

void f() {
  int flag = getInt();
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
  y = 1;
  flag = y;

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
  y = 1;
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

namespace condition_written_in_nested_stackframe_before_assignment {
int flag = 0;
int getInt();

void foo() {
  flag = getInt(); // tracking-note{{Value assigned to 'flag'}}
}

void f() {
  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}
  int y = 0;

  foo(); // tracking-note{{Calling 'foo'}}
         // tracking-note@-1{{Returning from 'foo'}}
  y = flag;

  if (y)    // expected-note{{Assuming 'y' is not equal to 0}}
            // expected-note@-1{{Taking true branch}}
            // debug-note@-2{{Tracking condition 'y'}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace condition_written_in_nested_stackframe_before_assignment

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

  flag = getInt();
  assert(flag); // tracking-note{{Calling 'assert'}}
                // tracking-note@-1{{Returning from 'assert'}}

  if (flag) // expected-note{{'flag' is not equal to 0}}
            // expected-note@-1{{Taking true branch}}
            // debug-note@-2{{Tracking condition 'flag'}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace unimportant_write_before_collapse_point
