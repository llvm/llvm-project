// RUN: %clang_analyze_cc1 -Wno-unused -std=c++11 -analyzer-checker=core,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -Wno-unused -std=c++17 -analyzer-checker=core,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -Wno-unused -std=c++11 -analyzer-checker=core,debug.ExprInspection -DMOVES -verify %s
// RUN: %clang_analyze_cc1 -Wno-unused -std=c++17 -analyzer-checker=core,debug.ExprInspection -DMOVES -verify %s

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

template <typename T> struct AddressVector {
  T *buf[10];
  int len;

  AddressVector() : len(0) {}

  void push(T *t) {
    buf[len] = t;
    ++len;
  }
};

class C {
  AddressVector<C> &v;

public:
  C(AddressVector<C> &v) : v(v) { v.push(this); }
  ~C() { v.push(this); }

#ifdef MOVES
  C(C &&c) : v(c.v) { v.push(this); }
#endif

  // Note how return-statements prefer move-constructors when available.
  C(const C &c) : v(c.v) {
#ifdef MOVES
    clang_analyzer_checkInlined(false); // no-warning
#else
    v.push(this);
#endif
  } // no-warning
};

@interface NSObject {}
@end;
@interface Foo: NSObject {}
  -(C) make: (AddressVector<C> &)v;
@end

@implementation Foo
-(C) make: (AddressVector<C> &)v {
  return C(v);
}
@end

void testReturnByValueFromMessage(Foo *foo) {
  AddressVector<C> v;
  {
    const C &c = [foo make: v];
  }
  // 0. Construct the return value of -make (copy/move elided) and
  //    lifetime-extend it directly via reference 'c',
  // 1. Destroy the temporary lifetime-extended by 'c'.
  clang_analyzer_eval(v.len == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == v.buf[1]); // expected-warning{{TRUE}}
}
