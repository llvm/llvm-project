// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify
// expected-no-diagnostics

class NonCopyable {
  NonCopyable(const NonCopyable&);
};

void capture_by_ref(NonCopyable nc, NonCopyable &ncr) {
  int array[3];
  (void)[&nc] () -> void {};
  (void)[&ncr] () -> void {}; 
  (void)[&array] () -> void {};
}
