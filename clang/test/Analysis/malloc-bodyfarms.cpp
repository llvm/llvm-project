// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker core,unix -verify %s

namespace std {
typedef struct once_flag_s {
  int _M_once = 0;
} once_flag;

template <class Callable, class... Args>
void call_once(once_flag &o, Callable&& func, Args&&... args);
} // namespace std

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);

void callee() {}

void test_no_state_change_in_body_farm() {
  std::once_flag flag;
  call_once(flag, callee); // no-crash
  malloc(1);
} // expected-warning{{Potential memory leak}}

void test_no_state_change_in_body_farm_2() {
  void *p = malloc(1);
  std::once_flag flag;
  call_once(flag, callee); // no-crash
  p = 0;
} // expected-warning{{Potential leak of memory pointed to by 'p'}}
