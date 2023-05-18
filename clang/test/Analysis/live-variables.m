// RUN: %clang_analyze_cc1 -analyzer-checker=core -fobjc-arc -verify %s
// expected-no-diagnostics
@interface NSObject
@end
@interface NSString : NSObject
- (id)lastPathComponent;
@end
int getBool(void);
int *getPtr(void);
int foo(void) {
  int r = 0;
  NSString *filename = @"filename";
  for (int x = 0; x< 10; x++) {
    int *p = getPtr();
    // Liveness info is not computed correctly due to the following expression.
    // This happens due to CFG being special cased for short circuit operators.
    // Note, due to ObjC method call, the outermost logical operator is wrapped in ExprWithCleanups.
    // PR18159
    if ((p != 0) && (getBool()) && ([filename lastPathComponent]) && (getBool())) {
      r = *p; // no-warning
    }
  }
  return r;
}