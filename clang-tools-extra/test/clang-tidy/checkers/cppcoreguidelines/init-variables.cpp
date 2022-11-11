// RUN: %check_clang_tidy %s cppcoreguidelines-init-variables %t -- -- -fno-delayed-template-parsing -fexceptions
// CHECK-FIXES: {{^}}#include <math.h>

// Ensure that function declarations are not changed.
void some_func(int x, double d, bool b, const char *p);

// Ensure that function arguments are not changed
int identity_function(int x) {
  return x;
}

int do_not_modify_me;

static int should_not_be_initialized;
extern int should_not_be_initialized2;

typedef struct {
  int unaltered1;
  int unaltered2;
} UnusedStruct;

typedef int my_int_type;
#define MACRO_INT int
#define FULL_DECLARATION() int macrodecl;

template <typename T>
void template_test_function() {
  T t;
  int uninitialized;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable 'uninitialized' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: {{^}}  int uninitialized = 0;{{$}}
}

void init_unit_tests() {
  int x;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable 'x' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: {{^}}  int x = 0;{{$}}
  my_int_type myint;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: variable 'myint' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: {{^}}  my_int_type myint = 0;{{$}}

  MACRO_INT macroint;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'macroint' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: {{^}}  MACRO_INT macroint = 0;{{$}}
  FULL_DECLARATION();

  int x0 = 1, x1, x2 = 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: variable 'x1' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: {{^}}  int x0 = 1, x1 = 0, x2 = 2;{{$}}
  int y0, y1 = 1, y2;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable 'y0' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-MESSAGES: :[[@LINE-2]]:19: warning: variable 'y2' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: {{^}}  int y0 = 0, y1 = 1, y2 = 0;{{$}}
  int hasval = 42;

  float f;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: variable 'f' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: {{^}}  float f = NAN;{{$}}
  float fval = 85.0;
  double d;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: variable 'd' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: {{^}}  double d = NAN;{{$}}
  double dval = 99.0;

  bool b;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: variable 'b' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: {{^}}  bool b = false;{{$}}
  bool bval = true;

  const char *ptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: variable 'ptr' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: {{^}}  const char *ptr = nullptr;{{$}}
  const char *ptrval = "a string";

  UnusedStruct u;

  static int does_not_need_an_initializer;
  extern int does_not_need_an_initializer2;
  int parens(42);
  int braces{42};
}

template <typename RANGE>
void f(RANGE r) {
  for (char c : r) {
  }
}

void catch_variable_decl() {
  // Expect no warning given here.
  try {
  } catch (int X) {
  }
}

enum Color { Red,
             Green,
             Blue };

enum Car { Benz,
           BMW = 20,
           Audi = BMW + 2 };

enum Gender : char { Male,
                     Female };

enum class Direction { Up,
                       Down,
                       Left,
                       Right };

enum class Fruit : int { Apple,
                         Orange };

void uninitialized_enum() {
  Color color;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: variable 'color' is not initialized [cppcoreguidelines-init-variables]
  Car car;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable 'car' is not initialized [cppcoreguidelines-init-variables]
  Gender gender;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: variable 'gender' is not initialized [cppcoreguidelines-init-variables]
  Direction direction;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'direction' is not initialized [cppcoreguidelines-init-variables]
  Fruit fruit;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: variable 'fruit' is not initialized [cppcoreguidelines-init-variables]
}
