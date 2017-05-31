// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -x c++ -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -x c++ -std=c++11 -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -x objective-c -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -x objective-c++ -std=c++11 -verify %s

#ifdef __cplusplus
# define EXTERN_C extern "C"
#else
# define EXTERN_C extern
#endif

EXTERN_C int printf(const char *,...);

typedef enum { Constant = 0 } TestEnum;
// Note that in C, the type of 'Constant' is 'int'. In C++ it is 'TestEnum'.
// This is why we don't check for that in the expected output.

void test(TestEnum input) {
    printf("%d", input); // no-warning
    printf("%d", Constant); // no-warning

    printf("%lld", input); // expected-warning-re{{format specifies type 'long long' but the argument has underlying type '{{(unsigned)?}} int'}}
    printf("%lld", Constant); // expected-warning{{format specifies type 'long long'}}
}


typedef enum { LongConstant = ~0UL } LongEnum;

void testLong(LongEnum input) {
  printf("%u", input); // expected-warning{{format specifies type 'unsigned int' but the argument has underlying type}}
  printf("%u", LongConstant); // expected-warning{{format specifies type 'unsigned int'}}
  
  printf("%lu", input);
  printf("%lu", LongConstant);
}
