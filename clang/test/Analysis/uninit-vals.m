// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store=region -verify %s
// expected-no-diagnostics

typedef unsigned int NSUInteger;

@interface A
- (NSUInteger)foo;
@end

NSUInteger f8(A* x){
  const NSUInteger n = [x foo];
  int* bogus;  

  if (n > 0) {    // tests const cast transfer function logic
    NSUInteger i;
    
    for (i = 0; i < n; ++i)
      bogus = 0;

    if (bogus)  // no-warning
      return n+1;
  }
  
  return n;
}


// PR10163 -- don't warn for default-initialized float arrays.
// (An additional test is in uninit-vals-ps-region.m)
void test_PR10163(float);
void PR10163 (void) {
  float x[2] = {0};
  test_PR10163(x[1]); // no-warning  
}
