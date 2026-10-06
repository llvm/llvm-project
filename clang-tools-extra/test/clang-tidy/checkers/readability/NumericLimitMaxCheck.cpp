// RUN: %check_clang_tidy %s readability-NumericLimitMaxCheck %t


typedef unsigned long long my_uint_t;

void test_arg(unsigned int);

void test_unsigned_literals() {
    
  unsigned int a = -1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use 'std::numeric_limits<unsigned int>::max()' instead of '-1' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: unsigned int a = std::numeric_limits<unsigned int>::max();

  unsigned int b = ~0;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use 'std::numeric_limits<unsigned int>::max()' instead of '~0' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: unsigned int b = std::numeric_limits<unsigned int>::max();

  unsigned long c = (unsigned long)(-1);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::numeric_limits<unsigned long>::max()' instead of '(unsigned long)(-1)' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: unsigned long c = std::numeric_limits<unsigned long>::max();

  unsigned long d = (unsigned long)(~0);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::numeric_limits<unsigned long>::max()' instead of '(unsigned long)(~0)' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: unsigned long d = std::numeric_limits<unsigned long>::max();

}

void test_comparisons(unsigned x) {
  if (x == -1) {
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use 'std::numeric_limits<unsigned int>::max()' instead of '-1' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: if (x == std::numeric_limits<unsigned int>::max()) {
    ;
  }

  if (x == (unsigned)(~0));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use 'std::numeric_limits<unsigned int>::max()' instead of '(unsigned)(~0)' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: if (x == std::numeric_limits<unsigned int>::max());

}

void test_cpp_casts_and_other_types() {
  unsigned int e = static_cast<unsigned int>(-1);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use 'std::numeric_limits<unsigned int>::max()' instead of 'static_cast<unsigned int>(-1)' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: unsigned int e = std::numeric_limits<unsigned int>::max();

  unsigned int f = unsigned(-1);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use 'std::numeric_limits<unsigned int>::max()' instead of 'unsigned(-1)' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: unsigned int f = std::numeric_limits<unsigned int>::max();

  unsigned short s = -1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use 'std::numeric_limits<unsigned short>::max()' instead of '-1' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: unsigned short s = std::numeric_limits<unsigned short>::max();

  unsigned char c = ~0;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::numeric_limits<unsigned char>::max()' instead of '~0' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: unsigned char c = std::numeric_limits<unsigned char>::max();
  
  // Tests that the check correctly uses the typedef'd name "my_uint_t"
  my_uint_t u = -1;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use 'std::numeric_limits<my_uint_t>::max()' instead of '-1' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: my_uint_t u = std::numeric_limits<my_uint_t>::max();
}

unsigned test_other_contexts(bool x) {
  // Test function arguments
  test_arg(-1);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use 'std::numeric_limits<unsigned int>::max()' instead of '-1' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: test_arg(std::numeric_limits<unsigned int>::max());

  // Test ternary operator
  unsigned k = x ? -1 : 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use 'std::numeric_limits<unsigned int>::max()' instead of '-1' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: unsigned k = x ? std::numeric_limits<unsigned int>::max() : 0;

  unsigned k2 = x ? 0 : ~0;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use 'std::numeric_limits<unsigned int>::max()' instead of '~0' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: unsigned k2 = x ? 0 : std::numeric_limits<unsigned int>::max();

  // Test return values
  return -1;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::numeric_limits<unsigned int>::max()' instead of '-1' [readability-NumericLimitMaxCheck]
  // CHECK-FIXES: return std::numeric_limits<unsigned int>::max();
}


#define MY_MAX_MACRO -1
#define MY_MAX_MACRO_TILDE ~0

void test_no_warning() {
  int a = -1;
  unsigned u = 0;
  int b = ~0;
  unsigned c = 42;
  
  // This is (max - 1), not max
  unsigned d = (unsigned)-2;

  // The check should ignore code from macros
  unsigned m = MY_MAX_MACRO;
  unsigned m2 = MY_MAX_MACRO_TILDE;
}
