// Opt-in gating for ObjC methods and blocks under an UNSPECIFIED default.
//
// With -fflow-sensitive-nullability on but no -fnullability-default and no
// explicit annotations, an ObjC method or block must NOT be analyzed — exactly
// like a plain C/C++ function. Otherwise flow-tracked nullability (e.g. a
// malloc() result) would warn on code that never opted in, holding ObjC/blocks
// to a stricter standard than functions. A method/block carrying an explicit
// nullability annotation opts in and is analyzed.
//
// RUN: %clang_cc1 -fsyntax-only -fblocks -fobjc-arc -fflow-sensitive-nullability -Wno-objc-root-class -Wno-unused-value %s -verify

typedef unsigned long size_t;
extern void *malloc(size_t);

@interface Gate
@end

@implementation Gate

// No annotations, unspecified default: not opted in -> not analyzed -> silent.
- (int)unannotated {
  int *p = (int *)malloc(4);
  return *p; // no warning: method did not opt in
}

// Explicit _Nullable on a parameter opts the method in -> malloc deref warns.
- (int)annotatedParam:(int *_Nullable)q {
  int *p = (int *)malloc(4);
  return *p; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// Explicit _Nullable on the return type opts the method in.
- (int *_Nullable)annotatedReturn {
  int *p = (int *)malloc(4);
  return p; // no deref; just proves the method is analyzed without crashing
}

@end

// Block with no annotations: not opted in -> silent.
void unannotated_block(void) {
  void (^b)(void) = ^{
    int *p = (int *)malloc(4);
    (void)*p; // no warning: block did not opt in
  };
  b();
}

// Block with an annotated parameter opts in -> malloc deref warns.
void annotated_block(void) {
  void (^b)(int *_Nullable) = ^(int *_Nullable q) {
    int *p = (int *)malloc(4);
    (void)*p; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  };
  b(0);
}
