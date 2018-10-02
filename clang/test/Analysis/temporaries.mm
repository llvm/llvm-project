// RUN: %clang_analyze_cc1 -analyzer-checker core,cplusplus -verify %s

// expected-no-diagnostics

#define nil ((id)0)

// Stripped down unique_ptr<int>
struct IntPtr {
  IntPtr(): i(new int) {}
  IntPtr(IntPtr &&o): i(o.i) { o.i = nullptr; }
  ~IntPtr() { delete i; }

  int *i;
};

@interface Foo {}
  -(void) foo: (IntPtr)arg;
@end

void testArgumentRegionInvalidation(Foo *f) {
  IntPtr ptr;
  int *i = ptr.i;
  [f foo: static_cast<IntPtr &&>(ptr)];
  *i = 99; // no-warning
}

void testNilReceiverCleanup() {
  [nil foo: IntPtr()];
}
