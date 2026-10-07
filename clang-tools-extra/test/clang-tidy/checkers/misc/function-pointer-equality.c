// RUN: %check_clang_tidy %s misc-function-pointer-equality %t

#define NULL ((void *)0)

void f(void);
void g(void);

typedef void (*fn_ptr)(void);

int test_c_equality(fn_ptr fp1, fn_ptr fp2) {
  if (f == g)
    return 1;
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  if (fp1 != fp2)
    return 2;
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]

  if (f == (void *) 0 || fp1 == 0)
    return 3;

  if (f == (void *) 1 || fp1 == ((void *) -1))
    return 3;

  return 0;
}
