// RUN: %clang_cc1 -triple x86_64-windows -fms-extensions -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-windows -fms-extensions -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -triple x86_64-windows -fms-extensions -fborland-extensions -fsyntax-only -verify %s

void outside_finally(void) {
  (void)__abnormal_termination(); // expected-error{{only allowed in __finally block}}
  (void)_abnormal_termination();  // expected-error{{only allowed in __finally block}}
}

void outside_finally_handlers(void) {
  __try {
    (void)__abnormal_termination(); // expected-error{{only allowed in __finally block}}
  } __except (1) {
    (void)__abnormal_termination(); // expected-error{{only allowed in __finally block}}
  }
}

void inside_finally(void) {
  __try {
  } __finally {
    (void)__abnormal_termination();
    (void)_abnormal_termination();
  }
}

#ifdef __cplusplus
void lambda_inside_finally(void) {
  __try {
  } __finally {
    auto L = [] { return __abnormal_termination(); }; // expected-error{{only allowed in __finally block}}
    (void)L();
  }
}
#endif
