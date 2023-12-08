// RUN: %clang_analyze_cc1 %s -std=c++17 \
// RUN:   -verify=expected,tracking \
// RUN:   -analyzer-config track-conditions=true \
// RUN:   -analyzer-output=text \
// RUN:   -analyzer-checker=core

// RUN: not %clang_analyze_cc1 -std=c++17 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config track-conditions=false \
// RUN:   -analyzer-config track-conditions-debug=true \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-DEBUG

// CHECK-INVALID-DEBUG: (frontend): invalid input for analyzer-config option
// CHECK-INVALID-DEBUG-SAME:        'track-conditions-debug', that expects
// CHECK-INVALID-DEBUG-SAME:        'track-conditions' to also be enabled
//
// RUN: %clang_analyze_cc1 %s -std=c++17 \
// RUN:   -verify=expected,tracking,debug \
// RUN:   -analyzer-config track-conditions=true \
// RUN:   -analyzer-config track-conditions-debug=true \
// RUN:   -analyzer-output=text \
// RUN:   -analyzer-checker=core

// RUN: %clang_analyze_cc1 %s -std=c++17 -verify \
// RUN:   -analyzer-output=text \
// RUN:   -analyzer-config track-conditions=false \
// RUN:   -analyzer-checker=core

namespace example_1 {
int flag;
bool coin();

void foo() {
  flag = coin(); // tracking-note-re{{{{^}}Value assigned to 'flag', which participates in a condition later{{$}}}}
}

void test() {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}
  flag = 1;

  foo(); // TODO: Add nodes here about flag's value being invalidated.
  if (flag) // expected-note-re   {{{{^}}Assuming 'flag' is 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking false branch{{$}}}}
    x = new int;

  foo(); // tracking-note-re{{{{^}}Calling 'foo'{{$}}}}
         // tracking-note-re@-1{{{{^}}Returning from 'foo'{{$}}}}

  if (flag) // expected-note-re   {{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}

    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace example_1

namespace example_2 {
int flag;
bool coin();

void foo() {
  flag = coin(); // tracking-note-re{{{{^}}Value assigned to 'flag', which participates in a condition later{{$}}}}
}

void test() {
  int *x = 0;
  flag = 1;

  foo();
  if (flag) // expected-note-re   {{{{^}}Assuming 'flag' is 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking false branch{{$}}}}
    x = new int;

  x = 0; // expected-note-re{{{{^}}Null pointer value stored to 'x'{{$}}}}

  foo(); // tracking-note-re{{{{^}}Calling 'foo'{{$}}}}
         // tracking-note-re@-1{{{{^}}Returning from 'foo'{{$}}}}

  if (flag) // expected-note-re   {{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}

    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace example_2

namespace global_variable_invalidation {
int flag;
bool coin();

void foo() {
  // coin() could write bar, do it's invalidated.
  flag = coin(); // tracking-note-re{{{{^}}Value assigned to 'flag', which participates in a condition later{{$}}}}
                 // tracking-note-re@-1{{{{^}}Value assigned to 'bar', which participates in a condition later{{$}}}}
}

int bar;

void test() {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}
  flag = 1;

  foo(); // tracking-note-re{{{{^}}Calling 'foo'{{$}}}}
         // tracking-note-re@-1{{{{^}}Returning from 'foo'{{$}}}}

  if (bar) // expected-note-re   {{{{^}}Assuming 'bar' is not equal to 0{{$}}}}
           // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
           // debug-note-re@-2{{{{^}}Tracking condition 'bar'{{$}}}}
    if (flag) // expected-note-re   {{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
              // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
              // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}

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
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  if (int flag = foo()) // debug-note-re{{{{^}}Tracking condition 'flag'{{$}}}}
                        // expected-note-re@-1{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
                        // expected-note-re@-2{{{{^}}Taking true branch{{$}}}}

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
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  if (ConvertsToBool())
    // expected-note-re@-1{{{{^}}Assuming the condition is true{{$}}}}
    // expected-note-re@-2{{{{^}}Taking true branch{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

} // namespace conversion_to_bool

namespace note_from_different_but_not_nested_stackframe {

void nullptrDeref(int *ptr, bool True) {
  if (True) // expected-note-re{{{{^}}'True' is true{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'True'{{$}}}}
    *ptr = 5;
  // expected-note@-1{{Dereference of null pointer (loaded from variable 'ptr')}}
  // expected-warning@-2{{Dereference of null pointer (loaded from variable 'ptr')}}
}

void f() {
  int *ptr = nullptr;
  // expected-note-re@-1{{{{^}}'ptr' initialized to a null pointer value{{$}}}}
  bool True = true;
  nullptrDeref(ptr, True);
  // expected-note-re@-1{{{{^}}Passing null pointer value via 1st parameter 'ptr'{{$}}}}
  // expected-note-re@-2{{{{^}}Calling 'nullptrDeref'{{$}}}}
}

} // end of namespace note_from_different_but_not_nested_stackframe

namespace important_returning_pointer_loaded_from {
bool coin();

int *getIntPtr();

void storeValue(int **i) {
  *i = getIntPtr();
}

int *conjurePointer() {
  int *i;
  storeValue(&i);
  return i;
}

void f(int *ptr) {
  if (ptr) // expected-note-re{{{{^}}Assuming 'ptr' is null{{$}}}}
           // expected-note-re@-1{{{{^}}Taking false branch{{$}}}}
    ;
  if (!conjurePointer())
    // expected-note-re@-1{{{{^}}Assuming the condition is true{{$}}}}
    // expected-note-re@-2{{{{^}}Taking true branch{{$}}}}
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
  if (ptr) // expected-note-re{{{{^}}Assuming 'ptr' is null{{$}}}}
           // expected-note-re@-1{{{{^}}Taking false branch{{$}}}}
    ;
  if (!conjurePointer())
    // expected-note-re@-1{{{{^}}Assuming the condition is true{{$}}}}
    // expected-note-re@-2{{{{^}}Taking true branch{{$}}}}
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
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  if (cast(conjure()))
    // expected-note-re@-1{{{{^}}Assuming the condition is false{{$}}}}
    // expected-note-re@-2{{{{^}}Taking false branch{{$}}}}
    return;
  *x = 5; // expected-warning{{Dereference of null pointer}}
          // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace unimportant_returning_pointer_loaded_from_through_cast

namespace unimportant_returning_value_note {
bool coin();

bool flipCoin() { return coin(); }

void i(int *ptr) {
  if (ptr) // expected-note-re{{{{^}}Assuming 'ptr' is null{{$}}}}
           // expected-note-re@-1{{{{^}}Taking false branch{{$}}}}
    ;
  if (!flipCoin())
    // expected-note-re@-1{{{{^}}Assuming the condition is true{{$}}}}
    // expected-note-re@-2{{{{^}}Taking true branch{{$}}}}
    *ptr = 5; // expected-warning{{Dereference of null pointer}}
              // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace unimportant_returning_value_note

namespace important_returning_value_note {
bool coin();

bool flipCoin() {
  if (coin())
    return true;
  return coin();
}

void i(int *ptr) {
  if (ptr) // expected-note-re{{{{^}}Assuming 'ptr' is null{{$}}}}
           // expected-note-re@-1{{{{^}}Taking false branch{{$}}}}
    ;
  if (!flipCoin())
    // expected-note-re@-1{{{{^}}Assuming the condition is true{{$}}}}
    // expected-note-re@-2{{{{^}}Taking true branch{{$}}}}
    *ptr = 5; // expected-warning{{Dereference of null pointer}}
              // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace important_returning_value_note

namespace important_returning_value_note_in_linear_function {
bool coin();
int flag;

struct super_complicated_template_hackery {
  static constexpr bool value = false;
};

void flipCoin() {
  if (super_complicated_template_hackery::value)
    // tracking-note-re@-1{{{{^}}'value' is false{{$}}}}
    // tracking-note-re@-2{{{{^}}Taking false branch{{$}}}}
    return;
  flag = false; // tracking-note-re{{{{^}}The value 0 is assigned to 'flag', which participates in a condition later{{$}}}}
}

void i(int *ptr) {
  flag = true;
  if (ptr) // expected-note-re{{{{^}}Assuming 'ptr' is null{{$}}}}
           // expected-note-re@-1{{{{^}}Taking false branch{{$}}}}
    ;
  flipCoin();
  // tracking-note-re@-1{{{{^}}Calling 'flipCoin'{{$}}}}
  // tracking-note-re@-2{{{{^}}Returning from 'flipCoin'{{$}}}}
  if (!flag)
    // debug-note-re@-1{{{{^}}Tracking condition '!flag'{{$}}}}
    // expected-note-re@-2{{{{^}}'flag' is 0{{$}}}}
    // expected-note-re@-3{{{{^}}Taking true branch{{$}}}}
    *ptr = 5; // expected-warning{{Dereference of null pointer}}
              // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace important_returning_value_note_in_linear_function

namespace tracked_condition_is_only_initialized {
int getInt();

void f() {
  int flag = getInt();
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}
  if (flag) // expected-note-re{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
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

  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}
  if (flag) // expected-note-re{{{{^}}'flag' is 1{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
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
  flag = y; // tracking-note-re{{{{^}}The value 1 is assigned to 'flag', which participates in a condition later{{$}}}}
}

void f(int y) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  foo(); // tracking-note-re{{{{^}}Calling 'foo'{{$}}}}
         // tracking-note-re@-1{{{{^}}Returning from 'foo'{{$}}}}

  if (flag) // expected-note-re{{{{^}}'flag' is 1{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace tracked_condition_written_in_nested_stackframe

namespace condition_written_in_nested_stackframe_before_assignment {
int flag = 0;
int getInt();

void foo() {
  flag = getInt(); // tracking-note-re{{{{^}}Value assigned to 'flag', which participates in a condition later{{$}}}}
}

void f() {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}
  int y = 0;

  foo(); // tracking-note-re{{{{^}}Calling 'foo'{{$}}}}
         // tracking-note-re@-1{{{{^}}Returning from 'foo'{{$}}}}
  y = flag;

  if (y)    // expected-note-re{{{{^}}Assuming 'y' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'y'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace condition_written_in_nested_stackframe_before_assignment

namespace dont_explain_foreach_loops {

struct Iterator {
  int *pos;
  bool operator!=(Iterator other) const {
    return pos && other.pos && pos != other.pos;
  }
  int operator*();
  Iterator operator++();
};

struct Container {
  Iterator begin();
  Iterator end();
};

void f(Container Cont) {
  int flag = 0;
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}
  for (int i : Cont)
    if (i) // expected-note-re   {{{{^}}Assuming 'i' is not equal to 0{{$}}}}
           // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
           // debug-note-re@-2{{{{^}}Tracking condition 'i'{{$}}}}
      flag = i;

  if (flag) // expected-note-re{{{{^}}'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace dont_explain_foreach_loops

namespace condition_lambda_capture_by_reference_last_write {
int getInt();

[[noreturn]] void halt();

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  auto lambda = [&flag]() {
    flag = getInt(); // tracking-note-re{{{{^}}Value assigned to 'flag', which participates in a condition later{{$}}}}
  };

  lambda(); // tracking-note-re{{{{^}}Calling 'operator()'{{$}}}}
            // tracking-note-re@-1{{{{^}}Returning from 'operator()'{{$}}}}

  if (flag) // expected-note-re{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace condition_lambda_capture_by_reference_last_write

namespace condition_lambda_capture_by_value_assumption {
int getInt();

[[noreturn]] void halt();

void bar(int &flag) {
  flag = getInt(); // tracking-note-re{{{{^}}Value assigned to 'flag', which participates in a condition later{{$}}}}
}

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  auto lambda = [flag]() {
    if (!flag) // tracking-note-re{{{{^}}Assuming 'flag' is not equal to 0, which participates in a condition later{{$}}}}
               // tracking-note-re@-1{{{{^}}Taking false branch{{$}}}}
      halt();
  };

  bar(flag); // tracking-note-re{{{{^}}Calling 'bar'{{$}}}}
             // tracking-note-re@-1{{{{^}}Returning from 'bar'{{$}}}}
  lambda();  // tracking-note-re{{{{^}}Calling 'operator()'{{$}}}}
             // tracking-note-re@-1{{{{^}}Returning from 'operator()'{{$}}}}

  if (flag) // expected-note-re{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace condition_lambda_capture_by_value_assumption

namespace condition_lambda_capture_by_reference_assumption {
int getInt();

[[noreturn]] void halt();

void bar(int &flag) {
  flag = getInt(); // tracking-note-re{{{{^}}Value assigned to 'flag', which participates in a condition later{{$}}}}
}

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  auto lambda = [&flag]() {
    if (!flag) // tracking-note-re{{{{^}}Assuming 'flag' is not equal to 0, which participates in a condition later{{$}}}}
               // tracking-note-re@-1{{{{^}}Taking false branch{{$}}}}
      halt();
  };

  bar(flag); // tracking-note-re{{{{^}}Calling 'bar'{{$}}}}
             // tracking-note-re@-1{{{{^}}Returning from 'bar'{{$}}}}
  lambda();  // tracking-note-re{{{{^}}Calling 'operator()'{{$}}}}
             // tracking-note-re@-1{{{{^}}Returning from 'operator()'{{$}}}}

  if (flag) // expected-note-re{{{{^}}'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace condition_lambda_capture_by_reference_assumption

namespace collapse_point_not_in_condition_bool {

[[noreturn]] void halt();

void check(bool b) {
  if (!b) // tracking-note-re{{{{^}}Assuming 'b' is true, which participates in a condition later{{$}}}}
          // tracking-note-re@-1{{{{^}}Taking false branch{{$}}}}
    halt();
}

void f(bool flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  check(flag); // tracking-note-re{{{{^}}Calling 'check'{{$}}}}
                // tracking-note-re@-1{{{{^}}Returning from 'check'{{$}}}}

  if (flag) // expected-note-re{{{{^}}'flag' is true{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace collapse_point_not_in_condition_bool

namespace collapse_point_not_in_condition {

[[noreturn]] void halt();

void assert(int b) {
  if (!b) // tracking-note-re{{{{^}}Assuming 'b' is not equal to 0, which participates in a condition later{{$}}}}
          // tracking-note-re@-1{{{{^}}Taking false branch{{$}}}}
    halt();
}

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  assert(flag); // tracking-note-re{{{{^}}Calling 'assert'{{$}}}}
                // tracking-note-re@-1{{{{^}}Returning from 'assert'{{$}}}}

  if (flag) // expected-note-re{{{{^}}'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace collapse_point_not_in_condition

namespace unimportant_write_before_collapse_point {

[[noreturn]] void halt();

void assert(int b) {
  if (!b) // tracking-note-re{{{{^}}Assuming 'b' is not equal to 0, which participates in a condition later{{$}}}}
          // tracking-note-re@-1{{{{^}}Taking false branch{{$}}}}
    halt();
}
int getInt();

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  flag = getInt();
  assert(flag); // tracking-note-re{{{{^}}Calling 'assert'{{$}}}}
                // tracking-note-re@-1{{{{^}}Returning from 'assert'{{$}}}}

  if (flag) // expected-note-re{{{{^}}'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace unimportant_write_before_collapse_point

namespace dont_crash_on_nonlogical_binary_operator {

void f6(int x) {
  int a[20];
  if (x == 25) {} // expected-note{{Assuming 'x' is equal to 25}}
                  // expected-note@-1{{Taking true branch}}
  if (a[x] == 123) {} // expected-warning{{The left operand of '==' is a garbage value due to array index out of bounds}}
                      // expected-note@-1{{The left operand of '==' is a garbage value due to array index out of bounds}}
}

} // end of namespace dont_crash_on_nonlogical_binary_operator

namespace collapse_point_not_in_condition_binary_op {

[[noreturn]] void halt();

void check(int b) {
  if (b == 1) // tracking-note-re{{{{^}}Assuming 'b' is not equal to 1, which participates in a condition later{{$}}}}
              // tracking-note-re@-1{{{{^}}Taking false branch{{$}}}}
    halt();
}

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  check(flag); // tracking-note-re{{{{^}}Calling 'check'{{$}}}}
               // tracking-note-re@-1{{{{^}}Returning from 'check'{{$}}}}

  if (flag) // expected-note-re{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace collapse_point_not_in_condition_binary_op

namespace collapse_point_not_in_condition_as_field {

[[noreturn]] void halt();
struct IntWrapper {
  int b;
  IntWrapper();

  void check() {
    if (!b) // tracking-note-re{{{{^}}Assuming field 'b' is not equal to 0, which participates in a condition later{{$}}}}
            // tracking-note-re@-1{{{{^}}Taking false branch{{$}}}}
      halt();
    return;
  }
};

void f(IntWrapper i) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  i.check(); // tracking-note-re{{{{^}}Calling 'IntWrapper::check'{{$}}}}
             // tracking-note-re@-1{{{{^}}Returning from 'IntWrapper::check'{{$}}}}
  if (i.b)   // expected-note-re{{{{^}}Field 'b' is not equal to 0{{$}}}}
             // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
             // debug-note-re@-2{{{{^}}Tracking condition 'i.b'{{$}}}}
    *x = 5;  // expected-warning{{Dereference of null pointer}}
             // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace collapse_point_not_in_condition_as_field

namespace assignemnt_in_condition_in_nested_stackframe {
int flag;

bool coin();

[[noreturn]] void halt();

void foo() {
  if ((flag = coin()))
    // tracking-note-re@-1{{{{^}}Value assigned to 'flag', which participates in a condition later{{$}}}}
    // tracking-note-re@-2{{{{^}}Assuming 'flag' is not equal to 0, which participates in a condition later{{$}}}}
    // tracking-note-re@-3{{{{^}}Taking true branch{{$}}}}
    return;
  halt();
  return;
}

void f() {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  foo();    // tracking-note-re{{{{^}}Calling 'foo'{{$}}}}
            // tracking-note-re@-1{{{{^}}Returning from 'foo'{{$}}}}
  if (flag) // expected-note-re{{{{^}}'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace assignemnt_in_condition_in_nested_stackframe

namespace condition_variable_less {
int flag;

bool coin();

[[noreturn]] void halt();

void foo() {
  if (flag > 0)
    // tracking-note-re@-1{{{{^}}Assuming 'flag' is > 0, which participates in a condition later{{$}}}}
    // tracking-note-re@-2{{{{^}}Taking true branch{{$}}}}
    return;
  halt();
  return;
}

void f() {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  foo();    // tracking-note-re{{{{^}}Calling 'foo'{{$}}}}
            // tracking-note-re@-1{{{{^}}Returning from 'foo'{{$}}}}
  if (flag) // expected-note-re{{{{^}}'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}
} // end of namespace condition_variable_less

namespace dont_track_assertlike_conditions {

extern void __assert_fail(__const char *__assertion, __const char *__file,
                          unsigned int __line, __const char *__function)
    __attribute__((__noreturn__));
#define assert(expr) \
  ((expr) ? (void)(0) : __assert_fail(#expr, __FILE__, __LINE__, __func__))

int getInt();

int cond1;

void bar() {
  cond1 = getInt();
}

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  flag = getInt();

  bar();
  assert(cond1); // expected-note-re{{{{^}}Assuming 'cond1' is not equal to 0{{$}}}}
                 // expected-note-re@-1{{{{^}}'?' condition is true{{$}}}}

  if (flag) // expected-note-re{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

#undef assert
} // end of namespace dont_track_assertlike_conditions

namespace dont_track_assertlike_and_conditions {

extern void __assert_fail(__const char *__assertion, __const char *__file,
                          unsigned int __line, __const char *__function)
    __attribute__((__noreturn__));
#define assert(expr) \
  ((expr) ? (void)(0) : __assert_fail(#expr, __FILE__, __LINE__, __func__))

int getInt();

int cond1;
int cond2;

void bar() {
  cond1 = getInt();
  cond2 = getInt();
}

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  flag = getInt();

  bar();
  assert(cond1 && cond2);
  // expected-note-re@-1{{{{^}}Assuming 'cond1' is not equal to 0{{$}}}}
  // expected-note-re@-2{{{{^}}Assuming 'cond2' is not equal to 0{{$}}}}
  // expected-note-re@-3{{{{^}}'?' condition is true{{$}}}}
  // expected-note-re@-4{{{{^}}Left side of '&&' is true{{$}}}}

  if (flag) // expected-note-re{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

#undef assert
} // end of namespace dont_track_assertlike_and_conditions

namespace dont_track_assertlike_or_conditions {

extern void __assert_fail(__const char *__assertion, __const char *__file,
                          unsigned int __line, __const char *__function)
    __attribute__((__noreturn__));
#define assert(expr) \
  ((expr) ? (void)(0) : __assert_fail(#expr, __FILE__, __LINE__, __func__))

int getInt();

int cond1;
int cond2;

void bar() {
  cond1 = getInt();
  cond2 = getInt();
}

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  flag = getInt();

  bar();
  assert(cond1 || cond2);
  // expected-note-re@-1{{{{^}}Assuming 'cond1' is not equal to 0{{$}}}}
  // expected-note-re@-2{{{{^}}Left side of '||' is true{{$}}}}

  if (flag) // expected-note-re{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

#undef assert
} // end of namespace dont_track_assertlike_or_conditions

namespace dont_track_assert2like_conditions {

extern void __assert_fail(__const char *__assertion, __const char *__file,
                          unsigned int __line, __const char *__function)
    __attribute__((__noreturn__));
#define assert(expr)                                      \
  do {                                                    \
    if (!(expr))                                          \
      __assert_fail(#expr, __FILE__, __LINE__, __func__); \
  } while (0)

int getInt();

int cond1;

void bar() {
  cond1 = getInt();
}

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  flag = getInt();

  bar();
  assert(cond1); // expected-note-re{{{{^}}Assuming 'cond1' is not equal to 0{{$}}}}
                 // expected-note-re@-1{{{{^}}Taking false branch{{$}}}}
                 // expected-note-re@-2{{{{^}}Loop condition is false.  Exiting loop{{$}}}}

  if (flag) // expected-note-re{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

#undef assert
} // end of namespace dont_track_assert2like_conditions

namespace dont_track_assert2like_and_conditions {

extern void __assert_fail(__const char *__assertion, __const char *__file,
                          unsigned int __line, __const char *__function)
    __attribute__((__noreturn__));
#define assert(expr)                                      \
  do {                                                    \
    if (!(expr))                                          \
      __assert_fail(#expr, __FILE__, __LINE__, __func__); \
  } while (0)

int getInt();

int cond1;
int cond2;

void bar() {
  cond1 = getInt();
  cond2 = getInt();
}

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  flag = getInt();

  bar();
  assert(cond1 && cond2);
  // expected-note-re@-1{{{{^}}Assuming 'cond1' is not equal to 0{{$}}}}
  // expected-note-re@-2{{{{^}}Left side of '&&' is true{{$}}}}
  // expected-note-re@-3{{{{^}}Assuming the condition is false{{$}}}}
  // expected-note-re@-4{{{{^}}Taking false branch{{$}}}}
  // expected-note-re@-5{{{{^}}Loop condition is false.  Exiting loop{{$}}}}

  if (flag) // expected-note-re{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

#undef assert
} // end of namespace dont_track_assert2like_and_conditions

namespace dont_track_assert2like_or_conditions {

extern void __assert_fail(__const char *__assertion, __const char *__file,
                          unsigned int __line, __const char *__function)
    __attribute__((__noreturn__));
#define assert(expr)                                      \
  do {                                                    \
    if (!(expr))                                          \
      __assert_fail(#expr, __FILE__, __LINE__, __func__); \
  } while (0)

int getInt();

int cond1;
int cond2;

void bar() {
  cond1 = getInt();
  cond2 = getInt();
}

void f(int flag) {
  int *x = 0; // expected-note-re{{{{^}}'x' initialized to a null pointer value{{$}}}}

  flag = getInt();

  bar();
  assert(cond1 || cond2);
  // expected-note-re@-1{{{{^}}Assuming 'cond1' is not equal to 0{{$}}}}
  // expected-note-re@-2{{{{^}}Left side of '||' is true{{$}}}}
  // expected-note-re@-3{{{{^}}Taking false branch{{$}}}}
  // expected-note-re@-4{{{{^}}Loop condition is false.  Exiting loop{{$}}}}

  if (flag) // expected-note-re{{{{^}}Assuming 'flag' is not equal to 0{{$}}}}
            // expected-note-re@-1{{{{^}}Taking true branch{{$}}}}
            // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    *x = 5; // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
}

#undef assert
} // end of namespace dont_track_assert2like_or_conditions

namespace only_track_the_evaluated_condition {

bool coin();

void bar(int &flag) {
  flag = coin(); // tracking-note-re{{{{^}}Value assigned to 'flag', which participates in a condition later{{$}}}}
}

void bar2(int &flag2) {
  flag2 = coin();
}

void f(int *x) {
  if (x) // expected-note-re{{{{^}}Assuming 'x' is null{{$}}}}
         // debug-note-re@-1{{{{^}}Tracking condition 'x'{{$}}}}
         // expected-note-re@-2{{{{^}}Taking false branch{{$}}}}
    return;

  int flag, flag2;
  bar(flag); // tracking-note-re{{{{^}}Calling 'bar'{{$}}}}
             // tracking-note-re@-1{{{{^}}Returning from 'bar'{{$}}}}
  bar2(flag2);

  if (flag && flag2) // expected-note-re   {{{{^}}Assuming 'flag' is 0{{$}}}}
                     // expected-note-re@-1{{{{^}}Left side of '&&' is false{{$}}}}
                     // debug-note-re@-2{{{{^}}Tracking condition 'flag'{{$}}}}
    return;

  *x = 5; // expected-warning{{Dereference of null pointer}}
          // expected-note@-1{{Dereference of null pointer}}
}

} // end of namespace only_track_the_evaluated_condition

namespace operator_call_in_condition_point {

struct Error {
  explicit operator bool() {
    return true;
  }
};

Error couldFail();

void f(int *x) {
  x = nullptr;              // expected-note {{Null pointer value stored to 'x'}}
  if (auto e = couldFail()) // expected-note {{Taking true branch}}
    *x = 5;                 // expected-warning {{Dereference of null pointer (loaded from variable 'x') [core.NullDereference]}}
                            // expected-note@-1 {{Dereference}}
}

} // namespace operator_call_in_condition_point

namespace cxx17_ifinit__operator_call_in_condition_point {

struct Error {
  explicit operator bool() {
    return true;
  }
};

Error couldFail();

void f(int *x) {
  x = nullptr;              // expected-note {{Null pointer value stored to 'x'}}
  if (auto e = couldFail(); e) // expected-note {{Taking true branch}}
    *x = 5;                 // expected-warning {{Dereference of null pointer (loaded from variable 'x') [core.NullDereference]}}
                            // expected-note@-1 {{Dereference}}
}

} // namespace cxx17_ifinit__operator_call_in_condition_point

namespace funcion_call_in_condition_point {

int alwaysTrue() {
  return true;
}

void f(int *x) {
  x = nullptr;      // expected-note {{Null pointer value stored to 'x'}}
  if (alwaysTrue()) // expected-note {{Taking true branch}}
    *x = 5;         // expected-warning {{Dereference of null pointer (loaded from variable 'x') [core.NullDereference]}}
                    // expected-note@-1 {{Dereference}}
}

} // namespace funcion_call_in_condition_point

namespace funcion_call_negated_in_condition_point {

int alwaysFalse() {
  return false;
}

void f(int *x) {
  x = nullptr;        // expected-note {{Null pointer value stored to 'x'}}
  if (!alwaysFalse()) // expected-note {{Taking true branch}}
    *x = 5;           // expected-warning {{Dereference of null pointer (loaded from variable 'x') [core.NullDereference]}}
                      // expected-note@-1 {{Dereference}}
}

} // namespace funcion_call_negated_in_condition_point

namespace funcion_call_part_of_logical_expr_in_condition_point {

int alwaysFalse() {
  return false;
}

void f(int *x) {
  x = nullptr;        // expected-note {{Null pointer value stored to 'x'}}
  if (!alwaysFalse() && true) // expected-note {{Taking true branch}}
                              // expected-note@-1 {{Left side of '&&' is true}}
    *x = 5;           // expected-warning {{Dereference of null pointer (loaded from variable 'x') [core.NullDereference]}}
                      // expected-note@-1 {{Dereference}}
}

} // namespace funcion_call_part_of_logical_expr_in_condition_point
