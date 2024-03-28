// REQUIRES: lld-available

// RUN: rm -rf %t && split-file %s %t && cd %t
//
// RUN: %clangxx_pgogen -fuse-ld=lld -O2 -g -fprofile-generate=. -mllvm -enable-vtable-value-profiling test.cpp -o test
// RUN: env LLVM_PROFILE_FILE=test.profraw ./test

// Show vtable profiles from raw profile.
// RUN: llvm-profdata show --function=main --ic-targets -show-vtables test.profraw | FileCheck %s --check-prefixes=COMMON,RAW

// Generate indexed profile from raw profile and show the data.
// RUN: llvm-profdata merge test.profraw -o test.profdata
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables test.profdata | FileCheck %s --check-prefixes=COMMON,INDEXED

// Generate text profile from raw and indexed profiles respectively and show the data.
// RUN: llvm-profdata merge --text test.profraw -o raw.proftext
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables --text raw.proftext | FileCheck %s --check-prefix=ICTEXT
// RUN: llvm-profdata merge --text test.profdata -o indexed.proftext
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables --text indexed.proftext | FileCheck %s --check-prefix=ICTEXT

// Generate indexed profile from text profiles and show the data
// RUN: llvm-profdata merge --binary raw.proftext -o text.profraw
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables text.profraw | FileCheck %s --check-prefixes=COMMON,INDEXED
// RUN: llvm-profdata merge --binary indexed.proftext -o text.profdata
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables text.profdata | FileCheck %s --check-prefixes=COMMON,INDEXED

// COMMON: Counters:
// COMMON-NEXT:  main:
// COMMON-NEXT:  Hash: 0x0f9a16fe6d398548
// COMMON-NEXT:  Counters: 2
// COMMON-NEXT:  Indirect Call Site Count: 2
// COMMON-NEXT:  Number of instrumented vtables: 2
// RAW:  Indirect Target Results:
// RAW-NEXT:       [  0, _ZN8Derived15func1Eii,        250 ] (25.00%)
// RAW-NEXT:       [  0, test.cpp;_ZN12_GLOBAL__N_18Derived25func1Eii,        750 ] (75.00%)
// RAW-NEXT:       [  1, _ZN8Derived15func2Eii,        250 ] (25.00%)
// RAW-NEXT:       [  1, test.cpp;_ZN12_GLOBAL__N_18Derived25func2Eii,        750 ] (75.00%)
// RAW-NEXT:  VTable Results:
// RAW-NEXT:       [  0, _ZTV8Derived1,        250 ] (25.00%)
// RAW-NEXT:       [  0, test.cpp;_ZTVN12_GLOBAL__N_18Derived2E,        750 ] (75.00%)
// RAW-NEXT:       [  1, _ZTV8Derived1,        250 ] (25.00%)
// RAW-NEXT:       [  1, test.cpp;_ZTVN12_GLOBAL__N_18Derived2E,        750 ] (75.00%)
// INDEXED:     Indirect Target Results:
// INDEXED-NEXT:         [  0, test.cpp;_ZN12_GLOBAL__N_18Derived25func1Eii,        750 ] (75.00%)
// INDEXED-NEXT:         [  0, _ZN8Derived15func1Eii,        250 ] (25.00%)
// INDEXED-NEXT:         [  1, test.cpp;_ZN12_GLOBAL__N_18Derived25func2Eii,        750 ] (75.00%)
// INDEXED-NEXT:         [  1, _ZN8Derived15func2Eii,        250 ] (25.00%)
// INDEXED-NEXT:     VTable Results:
// INDEXED-NEXT:         [  0, test.cpp;_ZTVN12_GLOBAL__N_18Derived2E,        750 ] (75.00%)
// INDEXED-NEXT:         [  0, _ZTV8Derived1,        250 ] (25.00%)
// INDEXED-NEXT:         [  1, test.cpp;_ZTVN12_GLOBAL__N_18Derived2E,        750 ] (75.00%)
// INDEXED-NEXT:         [  1, _ZTV8Derived1,        250 ] (25.00%)
// COMMON: Instrumentation level: IR  entry_first = 0
// COMMON-NEXT: Functions shown: 1
// COMMON-NEXT: Total functions: 6
// COMMON-NEXT: Maximum function count: 1000
// COMMON-NEXT: Maximum internal block count: 250
// COMMON-NEXT: Statistics for indirect call sites profile:
// COMMON-NEXT:   Total number of sites: 2
// COMMON-NEXT:   Total number of sites with values: 2
// COMMON-NEXT:   Total number of profiled values: 4
// COMMON-NEXT:   Value sites histogram:
// COMMON-NEXT:         NumTargets, SiteCount
// COMMON-NEXT:         2, 2
// COMMON-NEXT: Statistics for vtable profile:
// COMMON-NEXT:   Total number of sites: 2
// COMMON-NEXT:   Total number of sites with values: 2
// COMMON-NEXT:   Total number of profiled values: 4
// COMMON-NEXT:   Value sites histogram:
// COMMON-NEXT:         NumTargets, SiteCount
// COMMON-NEXT:         2, 2

// ICTEXT: :ir
// ICTEXT: main
// ICTEXT: # Func Hash:
// ICTEXT: 1124236338992350536
// ICTEXT: # Num Counters:
// ICTEXT: 2
// ICTEXT: # Counter Values:
// ICTEXT: 1000
// ICTEXT: 1
// ICTEXT: # Num Value Kinds:
// ICTEXT: 2
// ICTEXT: # ValueKind = IPVK_IndirectCallTarget:
// ICTEXT: 0
// ICTEXT: # NumValueSites:
// ICTEXT: 2
// ICTEXT: 2
// ICTEXT: test.cpp;_ZN12_GLOBAL__N_18Derived25func1Eii:750
// ICTEXT: _ZN8Derived15func1Eii:250
// ICTEXT: 2
// ICTEXT: test.cpp;_ZN12_GLOBAL__N_18Derived25func2Eii:750
// ICTEXT: _ZN8Derived15func2Eii:250
// ICTEXT: # ValueKind = IPVK_VTableTarget:
// ICTEXT: 2
// ICTEXT: # NumValueSites:
// ICTEXT: 2
// ICTEXT: 2
// ICTEXT: test.cpp;_ZTVN12_GLOBAL__N_18Derived2E:750
// ICTEXT: _ZTV8Derived1:250
// ICTEXT: 2
// ICTEXT: test.cpp;_ZTVN12_GLOBAL__N_18Derived2E:750
// ICTEXT: _ZTV8Derived1:250

//--- test.cpp
#include <cstdio>
#include <cstdlib>
class Base {
public:
  virtual int func1(int a, int b) = 0;
  virtual int func2(int a, int b) = 0;
};
class Derived1 : public Base {
public:
  int func1(int a, int b) override { return a + b; }

  int func2(int a, int b) override { return a * b; }
};
namespace {
class Derived2 : public Base {
public:
  int func1(int a, int b) override { return a - b; }

  int func2(int a, int b) override { return a * (a - b); }
};
} // namespace
__attribute__((noinline)) Base *createType(int a) {
  Base *base = nullptr;
  if (a % 4 == 0)
    base = new Derived1();
  else
    base = new Derived2();
  return base;
}
int main(int argc, char **argv) {
  int sum = 0;
  for (int i = 0; i < 1000; i++) {
    int a = rand();
    int b = rand();
    Base *ptr = createType(i);
    sum += ptr->func1(a, b) + ptr->func2(b, a);
  }
  printf("sum is %d\n", sum);
  return 0;
}
