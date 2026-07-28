// RUN: %clang_cc1 -std=c23 -fobjc-arc -verify %s

typedef struct {
  id a;
} S;

void test1(void) {
  (void)(thread_local static __unsafe_unretained id){0};
  (void)(thread_local static id){0}; // expected-error {{thread-local variable has non-trivial ownership: type is '__strong id'}}
  (void)(thread_local static S){0};  // expected-error {{type of thread-local variable has non-trivial destruction}}
}
