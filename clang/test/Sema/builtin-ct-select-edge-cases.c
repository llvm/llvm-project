// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter

// Test with various condition expressions
int test_conditional_expressions(int x, int y, int a, int b) {
  // Logical expressions
  int result1 = __builtin_ct_select(x && y, a, b);
  int result2 = __builtin_ct_select(x || y, a, b);
  int result3 = __builtin_ct_select(!x, a, b);
  
  // Comparison expressions
  int result4 = __builtin_ct_select(x == y, a, b);
  int result5 = __builtin_ct_select(x != y, a, b);
  int result6 = __builtin_ct_select(x < y, a, b);
  int result7 = __builtin_ct_select(x > y, a, b);
  int result8 = __builtin_ct_select(x <= y, a, b);
  int result9 = __builtin_ct_select(x >= y, a, b);
  
  // Bitwise expressions
  int result10 = __builtin_ct_select(x & y, a, b);
  int result11 = __builtin_ct_select(x | y, a, b);
  int result12 = __builtin_ct_select(x ^ y, a, b);
  int result13 = __builtin_ct_select(~x, a, b);
  
  // Arithmetic expressions
  int result14 = __builtin_ct_select(x + y, a, b);
  int result15 = __builtin_ct_select(x - y, a, b);
  int result16 = __builtin_ct_select(x * y, a, b);
  int result17 = __builtin_ct_select(x / y, a, b);
  int result18 = __builtin_ct_select(x % y, a, b);
  
  return result1 + result2 + result3 + result4 + result5 + result6 + result7 + result8 + result9 + result10 + result11 + result12 + result13 + result14 + result15 + result16 + result17 + result18;
}

// Test with extreme values
int test_extreme_values(int cond) {
  // Maximum and minimum values
  int max_int = __builtin_ct_select(cond, __INT_MAX__, -__INT_MAX__ - 1);
  
  // Very large numbers
  long long max_ll = __builtin_ct_select(cond, __LONG_LONG_MAX__, -__LONG_LONG_MAX__ - 1);
  
  // Floating point extremes
  float max_float = __builtin_ct_select(cond, __FLT_MAX__, -__FLT_MAX__);
  double max_double = __builtin_ct_select(cond, __DBL_MAX__, -__DBL_MAX__);
  
  return max_int;
}

// Test with zero and negative zero
int test_zero_values(int cond) {
  // Integer zeros
  int zero_int = __builtin_ct_select(cond, 0, -0);
  
  // Floating point zeros
  float zero_float = __builtin_ct_select(cond, 0.0f, -0.0f);
  double zero_double = __builtin_ct_select(cond, 0.0, -0.0);
  
  return zero_int;
}

// Test with infinity and NaN
int test_special_float_values(int cond) {
  // Infinity
  float inf_float = __builtin_ct_select(cond, __builtin_inff(), -__builtin_inff());
  double inf_double = __builtin_ct_select(cond, __builtin_inf(), -__builtin_inf());
  
  // NaN
  float nan_float = __builtin_ct_select(cond, __builtin_nanf(""), __builtin_nanf(""));
  double nan_double = __builtin_ct_select(cond, __builtin_nan(""), __builtin_nan(""));
  
  return 0;
}

// Test with complex pointer scenarios
int test_pointer_edge_cases(int cond) {
  int arr[10];
  int *ptr1 = arr;
  int *ptr2 = arr + 5;
  
  // Array pointers
  int *result1 = __builtin_ct_select(cond, ptr1, ptr2);
  
  // Pointer arithmetic
  int *result2 = __builtin_ct_select(cond, arr + 1, arr + 2);
  
  // NULL vs non-NULL
  int *result3 = __builtin_ct_select(cond, ptr1, (int*)0);
  
  // Different pointer types (should fail)
  float *fptr = (float*)0;
  int *result4 = __builtin_ct_select(cond, ptr1, fptr); // expected-error {{incompatible operand types ('int *' and 'float *')}}
  
  return *result1;
}

// Test with function pointers
int func1(int x) { return x; }
int func2(int x) { return x * 2; }
float func3(float x) { return x; }

int test_function_pointers(int cond, int x) {
  // Same signature function pointer 
  int (*fptr)(int) = __builtin_ct_select(cond, &func1, &func2);
  
  // Different signature function pointers (should fail)
  int (*bad_fptr)(int) = __builtin_ct_select(cond, &func1, &func3); // expected-error {{incompatible operand types ('int (*)(int)' and 'float (*)(float)')}}
  
  return fptr(x);
}

// Test with void pointers
void *test_void_pointers(int cond, void *a, void *b) {
  return __builtin_ct_select(cond, a, b);
}

// Test with const/volatile qualifiers
int test_qualifiers(int cond) {
  const int ca = 10;
  const int cb = 20;
  volatile int va = 30;
  volatile int vb = 40;
  const volatile int cva = 50;
  const volatile int cvb = 60;
  
  // const to const
  const int result1 = __builtin_ct_select(cond, ca, cb);
  
  // volatile to volatile
  volatile int result2 = __builtin_ct_select(cond, va, vb);
  
  // const volatile to const volatile
  const volatile int result3 = __builtin_ct_select(cond, cva, cvb);
  
  return result1 + result2 + result3;
}

// Test with arrays (should fail as they're not arithmetic or pointer)
int test_arrays(int cond) {
  int arr1[5] = {1, 2, 3, 4, 5};
  int arr2[5] = {6, 7, 8, 9, 10};
  
  // This should fail??
  int *result = __builtin_ct_select(cond, arr1, arr2); 
  
  return result[0];
}

// Test with structures (should fail)
struct Point {
  int x, y;
};

struct Point test_structs(int cond) {
  struct Point p1 = {1, 2};
  struct Point p2 = {3, 4};
  
  return __builtin_ct_select(cond, p1, p2); // expected-error {{incompatible operand types ('struct Point' and 'struct Point')}}
}

// Test with unions (should fail)
union Data {
  int i;
  float f;
};

union Data test_unions(int cond) {
  union Data d1 = {.i = 10};
  union Data d2 = {.i = 20};
  
  return __builtin_ct_select(cond, d1, d2); // expected-error {{incompatible operand types ('union Data' and 'union Data')}}
}

// Test with bit fields (should work as they're integers)
struct BitField {
  int a : 4;
  int b : 4;
};

int test_bit_fields(int cond) {
  struct BitField bf1 = {1, 2};
  struct BitField bf2 = {3, 4};
  
  // Individual bit fields should work
  int result1 = __builtin_ct_select(cond, bf1.a, bf2.a);
  int result2 = __builtin_ct_select(cond, bf1.b, bf2.b);
  
  return result1 + result2;
}

// Test with designated initializers
int test_designated_init(int cond) {
  int arr1[3] = {[0] = 1, [1] = 2, [2] = 3};
  int arr2[3] = {[0] = 4, [1] = 5, [2] = 6};
  
  // Access specific elements
  int result1 = __builtin_ct_select(cond, arr1[0], arr2[0]);
  int result2 = __builtin_ct_select(cond, arr1[1], arr2[1]);
  
  return result1 + result2;
}

// Test with complex expressions in arguments
int complex_expr(int x) { return x * x; }

int test_complex_arguments(int cond, int x, int y) {
  // Function calls as arguments
  int result1 = __builtin_ct_select(cond, complex_expr(x), complex_expr(y));
  
  // Ternary operator as arguments
  int result2 = __builtin_ct_select(cond, x > 0 ? x : -x, y > 0 ? y : -y);
  
  // Compound literals
  int result3 = __builtin_ct_select(cond, (int){x}, (int){y});
  
  return result1 + result2 + result3;
}

// Test with preprocessor macros
#define MACRO_A 42
#define MACRO_B 24
#define MACRO_COND(x) (x > 0)

int test_macros(int x) {
  int result1 = __builtin_ct_select(MACRO_COND(x), MACRO_A, MACRO_B);
  
  // Nested macros
  #define NESTED_SELECT(c, a, b) __builtin_ct_select(c, a, b)
  int result2 = NESTED_SELECT(x, 10, 20);
  
  return result1 + result2;
}

// Test with string literals (should fail)
const char *test_strings(int cond) {
  return __builtin_ct_select(cond, "hello", "world"); 
}

// Test with variable length arrays (VLA)
int test_vla(int cond, int n) {
  int vla1[n];
  int vla2[n];
  
  // Individual elements should work
  vla1[0] = 1;
  vla2[0] = 2;
  int result = __builtin_ct_select(cond, vla1[0], vla2[0]); 
  
  return result;
}

// Test with typedef
typedef int MyInt;
typedef float MyFloat;

MyInt test_typedef(int cond, MyInt a, MyInt b) {
  return __builtin_ct_select(cond, a, b);
}

// Test with different typedef types (should fail)
MyInt test_different_typedef(int cond, MyInt a, MyFloat b) {
  return __builtin_ct_select(cond, a, b); // expected-error {{incompatible operand types ('MyInt' (aka 'int') and 'MyFloat' (aka 'float'))}}
}

// Test with side effects (should be evaluated)
int side_effect_counter = 0;
int side_effect_func(int x) {
  side_effect_counter++;
  return x;
}

int test_side_effects(int cond) {
  // Both arguments should be evaluated
  int result = __builtin_ct_select(cond, side_effect_func(10), side_effect_func(20));
  return result;
}

// Test with goto labels (context where expressions are used)
int test_goto_context(int cond, int a, int b) {
  int result = __builtin_ct_select(cond, a, b);
  
  if (result > 0) {
    goto positive;
  } else {
    goto negative;
  }
  
positive:
  return result;
  
negative:
  return -result;
}

// Test with switch statements
int test_switch_context(int cond, int a, int b) {
  int result = __builtin_ct_select(cond, a, b);
  
  switch (result) {
    case 0:
      return 0;
    case 1:
      return 1;
    default:
      return -1;
  }
}

// Test with loops
int test_loop_context(int cond, int a, int b) {
  int result = __builtin_ct_select(cond, a, b);
  int sum = 0;
  
  for (int i = 0; i < result; i++) {
    sum += i;
  }
  
  return sum;
}

// Test with recursive functions
int factorial(int n) {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

int test_recursive(int cond, int n) {
  int result = __builtin_ct_select(cond, n, n + 1);
  return factorial(result);
}

// Test with inline functions
static inline int inline_func(int x) {
  return x * 2;
}

int test_inline(int cond, int a, int b) {
  return __builtin_ct_select(cond, inline_func(a), inline_func(b));
}

// Test with static variables
int test_static_vars(int cond) {
  static int static_a = 10;
  static int static_b = 20;
  
  return __builtin_ct_select(cond, static_a, static_b);
}

// Test with extern variables
extern int extern_a;
extern int extern_b;

int test_extern_vars(int cond) {
  return __builtin_ct_select(cond, extern_a, extern_b);
}

// Test with register variables
int test_register_vars(int cond) {
  register int reg_a = 30;
  register int reg_b = 40;
  
  return __builtin_ct_select(cond, reg_a, reg_b);
}

// Test with thread-local variables (C11)
#if __STDC_VERSION__ >= 201112L
_Thread_local int tls_a = 50;
_Thread_local int tls_b = 60;

int test_tls_vars(int cond) {
  return __builtin_ct_select(cond, tls_a, tls_b);
}
#endif
