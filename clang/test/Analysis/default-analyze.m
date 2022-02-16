// RUN: %clang_analyze_cc1 %s -o %t

// Tests that some specific checkers are enabled by default.

id foo(int x) {
  id title;
  switch (x) {
  case 1:
    title = @"foo"; // expected-warning {{never read}}
  case 2:
    title = @"bar";
    break;
  default:
    title = @"baz";
    break;
  }
  return title;
}

// <rdar://problem/8808566> Static analyzer is wrong: NSWidth(imgRect) not understood as unconditional assignment
//
// Note: this requires inlining support.  This previously issued a false positive use of
// uninitialized value when calling NSWidth.
typedef double CGFloat;

struct CGPoint {
  CGFloat x;
  CGFloat y;
};
typedef struct CGPoint CGPoint;

struct CGSize {
  CGFloat width;
  CGFloat height;
};
typedef struct CGSize CGSize;

struct CGRect {
  CGPoint origin;
  CGSize size;
};
typedef struct CGRect CGRect;

typedef CGRect NSRect;
typedef CGSize NSSize;

static __inline__ __attribute__((always_inline)) CGFloat NSWidth(NSRect aRect) {
    return (aRect.size.width);
}

static __inline__ __attribute__((always_inline)) CGFloat NSHeight(NSRect aRect) {
    return (aRect.size.height);
}

NSSize rdar880566_size(void);

double rdar8808566(void) {
  NSRect myRect;
  myRect.size = rdar880566_size();
  double x = NSWidth(myRect) + NSHeight(myRect); // no-warning
  return x;
}

