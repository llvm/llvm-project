// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-ipa=inlining -verify %s

void clang_analyzer_eval(bool);

typedef struct Opaque *Data;
struct IntWrapper {
  int x;
};

struct Child : public IntWrapper {
  void set() { x = 42; }
};

void test(Data data) {
  Child *wrapper = reinterpret_cast<Child*>(data);
  // Don't crash when upcasting here.
  // We don't actually know if 'data' is a Child.
  wrapper->set();
  clang_analyzer_eval(wrapper->x == 42); // expected-warning{{TRUE}}
}
