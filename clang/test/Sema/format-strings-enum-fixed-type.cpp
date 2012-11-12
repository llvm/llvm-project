// RUN: %clang_cc1 -triple i386-apple-darwin9 -x c++ -std=c++11 -verify %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -x objective-c -verify %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -x objective-c++ -verify %s

#ifdef __cplusplus
# define EXTERN_C extern "C"
#else
# define EXTERN_C extern
#endif

EXTERN_C int printf(const char *,...);

typedef enum : short { Constant = 0 } TestEnum;
// Note that in C (and Objective-C), the type of 'Constant' is 'short'.
// In C++ (and Objective-C++) it is 'TestEnum'.
// This is why we don't check for that in the expected output.

void test(TestEnum input) {
  printf("%hhd", input); // expected-warning{{format specifies type 'char' but the argument has type 'TestEnum'}}
  printf("%hhd", Constant); // expected-warning{{format specifies type 'char'}}
  
  printf("%hd", input); // no-warning
  printf("%hd", Constant); // no-warning

  // While these are less correct, they are still safe.
  printf("%d", input); // no-warning
  printf("%d", Constant); // no-warning
  
  printf("%lld", input); // expected-warning{{format specifies type 'long long' but the argument has type 'TestEnum'}}
  printf("%lld", Constant); // expected-warning{{format specifies type 'long long'}}
}


typedef enum : unsigned long { LongConstant = ~0UL } LongEnum;

void testLong(LongEnum input) {
  printf("%u", input); // expected-warning{{format specifies type 'unsigned int' but the argument has type 'LongEnum'}}
  printf("%u", LongConstant); // expected-warning{{format specifies type 'unsigned int'}}
  
  printf("%lu", input);
  printf("%lu", LongConstant);
}


typedef short short_t;
typedef enum : short_t { ShortConstant = 0 } ShortEnum;

void testUnderlyingTypedef(ShortEnum input) {
  printf("%hhd", input); // expected-warning{{format specifies type 'char' but the argument has type 'ShortEnum'}}
  printf("%hhd", ShortConstant); // expected-warning{{format specifies type 'char'}}
  
  printf("%hd", input); // no-warning
  printf("%hd", ShortConstant); // no-warning
  
  // While these are less correct, they are still safe.
  printf("%d", input); // no-warning
  printf("%d", ShortConstant); // no-warning
  
  printf("%lld", input); // expected-warning{{format specifies type 'long long' but the argument has type 'ShortEnum'}}
  printf("%lld", ShortConstant); // expected-warning{{format specifies type 'long long'}}
}


typedef ShortEnum ShortEnum2;

void testTypedefChain(ShortEnum2 input) {
  printf("%hhd", input); // expected-warning{{format specifies type 'char' but the argument has type 'ShortEnum2' (aka 'ShortEnum')}}
  printf("%hd", input); // no-warning
  printf("%d", input); // no-warning
  printf("%lld", input); // expected-warning{{format specifies type 'long long' but the argument has type 'ShortEnum2' (aka 'ShortEnum')}}
}


typedef enum : char { CharConstant = 'a' } CharEnum;

// %hhd is deliberately not required to be signed, because 'char' isn't either.
// This is a separate code path in FormatString.cpp.
void testChar(CharEnum input) {
  printf("%hhd", input); // no-warning
  printf("%hhd", CharConstant); // no-warning

  // This is not correct but it is safe. We warn because '%hd' shows intent.
  printf("%hd", input); // expected-warning{{format specifies type 'short' but the argument has type 'CharEnum'}}
  printf("%hd", CharConstant); // expected-warning{{format specifies type 'short'}}
  
  // This is not correct but it matches the promotion rules (and is safe).
  printf("%d", input); // no-warning
  printf("%d", CharConstant); // no-warning
  
  printf("%lld", input); // expected-warning{{format specifies type 'long long' but the argument has type 'CharEnum'}}
  printf("%lld", CharConstant); // expected-warning{{format specifies type 'long long'}}
}
