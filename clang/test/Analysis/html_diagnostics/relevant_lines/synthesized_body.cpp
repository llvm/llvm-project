// Faking std::call_once implementation.
namespace std {
typedef struct once_flag_s {
  int _M_once = 0;
} once_flag;

template <class Callable, class... Args>
void call_once(once_flag &o, Callable&& func, Args&&... args);
} // namespace std

int deref(int *x) {
  return *x;
}

void call_deref_once() {
  static std::once_flag once;
  int *p = nullptr;
  std::call_once(once, &deref, p);
}


// RUN: rm -rf %t.output
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core -analyzer-output html -o %t.output %s
// RUN: cat %t.output/* | FileCheck %s --match-full-lines
// CHECK: var relevant_lines = {"1": {"3": 1,  "8": 1, "11": 1, "12": 1, "15": 1, "16": 1, "17": 1, "18": 1}};
