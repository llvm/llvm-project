// RUN: %clang_analyze_cc1 -analyzer-checker=core %s -verify
// expected-no-diagnostics

//===-- unions-region.m ---------------------------------------------------===//
//
// This file tests the analyzer's reasoning about unions.
//
//===----------------------------------------------------------------------===//

// [testA] When using RegionStore, this test case previously had a
// false positive of a 'pass-by-value argument is uninitialized'
// warning at the call to 'testA_aux' and 'testA_aux_2'.
union u_testA {
  unsigned i;
  float f;
};
 
float testA(float f) {
  int testA_aux(unsigned x);
  int testA_aux_2(union u_testA z);
  
  union u_testA swap;
  swap.f = f;

  if (testA_aux(swap.i))  // no-warning
    swap.i = ((swap.i & 0xffff0000) >> 16) | ((swap.i & 0x0000fffff) << 16);

  testA_aux_2(swap); // no-warning

  return swap.f;  
}

// [testB] When using RegionStore, this test case previously had a
// false positive of a 'pass-by-value argument is uninitialized'
// warning at the call to 'testB_aux'.
void testB(int i) {
  void testB_aux(short z);
  union { short x[2]; unsigned y; } val;  
  val.y = 10;
  testB_aux(val.x[1]); // no-warning
}

