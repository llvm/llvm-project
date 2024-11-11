// REQUIRES: lld, lld-available

// Building the instrumented binary will fail because lld doesn't support
// big-endian ELF for PPC (aka ABI 1).
// ld.lld: error: /lib/../lib64/Scrt1.o: ABI version 1 is not supported
// UNSUPPORTED: ppc && host-byteorder-big-endian

// RUN: rm -rf %t && mkdir %t && split-file %s %t && cd %t

// RUN: %clangxx_pgogen -fuse-ld=lld -O2 -fprofile-generate=. -mllvm -enable-vtable-value-profiling lib.cpp main.cpp -o test
// RUN: env LLVM_PROFILE_FILE=test.profraw ./test

// Show vtable profiles from raw profile.
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables test.profraw | FileCheck %s --check-prefixes=COMMON,RAW

// Generate indexed profile from raw profile and show the data.
// RUN: llvm-profdata merge --keep-vtable-symbols test.profraw -o test.profdata
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables test.profdata | FileCheck %s --check-prefixes=COMMON,INDEXED

// Generate text profile from raw and indexed profiles respectively and show the data.
// RUN: llvm-profdata merge --keep-vtable-symbols --text test.profraw -o raw.proftext
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables --text raw.proftext | FileCheck %s --check-prefix=ICTEXT
// RUN: llvm-profdata merge --keep-vtable-symbols --text test.profdata -o indexed.proftext
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables --text indexed.proftext | FileCheck %s --check-prefix=ICTEXT

// Generate indexed profile from text profiles and show the data
// RUN: llvm-profdata merge --keep-vtable-symbols --binary raw.proftext -o text.profraw
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables text.profraw | FileCheck %s --check-prefixes=COMMON,INDEXED
// RUN: llvm-profdata merge --keep-vtable-symbols --binary indexed.proftext -o text.profdata
// RUN: llvm-profdata show --function=main --ic-targets --show-vtables text.profdata | FileCheck %s --check-prefixes=COMMON,INDEXED

// COMMON: Counters:
// COMMON-NEXT:  main:
// COMMON-NEXT:  Hash: 0x068617320ec408a0
// COMMON-NEXT:  Counters: 4
// COMMON-NEXT:  Indirect Call Site Count: 2
// COMMON-NEXT:  Number of instrumented vtables: 2
// RAW:  Indirect Target Results:
// RAW-NEXT:       [  0, _ZN8Derived14funcEii,        50 ] (25.00%)
// RAW-NEXT:       [  0, {{.*}}lib.cpp;_ZN12_GLOBAL__N_18Derived24funcEii,        150 ] (75.00%)
// RAW-NEXT:       [  1, _ZN8Derived1D0Ev,        250 ] (25.00%)
// RAW-NEXT:       [  1, {{.*}}lib.cpp;_ZN12_GLOBAL__N_18Derived2D0Ev,        750 ] (75.00%)
// RAW-NEXT:  VTable Results:
// RAW-NEXT:       [  0, _ZTV8Derived1,        50 ] (25.00%)
// RAW-NEXT:       [  0, {{.*}}lib.cpp;_ZTVN12_GLOBAL__N_18Derived2E,        150 ] (75.00%)
// RAW-NEXT:       [  1, _ZTV8Derived1,        250 ] (25.00%)
// RAW-NEXT:       [  1, {{.*}}lib.cpp;_ZTVN12_GLOBAL__N_18Derived2E,        750 ] (75.00%)
// INDEXED:     Indirect Target Results:
// INDEXED-NEXT:         [  0, {{.*}}lib.cpp;_ZN12_GLOBAL__N_18Derived24funcEii,        150 ] (75.00%)
// INDEXED-NEXT:         [  0, _ZN8Derived14funcEii,        50 ] (25.00%)
// INDEXED-NEXT:         [  1, {{.*}}lib.cpp;_ZN12_GLOBAL__N_18Derived2D0Ev,        750 ] (75.00%)
// INDEXED-NEXT:         [  1, _ZN8Derived1D0Ev,        250 ] (25.00%)
// INDEXED-NEXT:     VTable Results:
// INDEXED-NEXT:         [  0, {{.*}}lib.cpp;_ZTVN12_GLOBAL__N_18Derived2E,        150 ] (75.00%)
// INDEXED-NEXT:         [  0, _ZTV8Derived1,        50 ] (25.00%)
// INDEXED-NEXT:         [  1, {{.*}}lib.cpp;_ZTVN12_GLOBAL__N_18Derived2E,        750 ] (75.00%)
// INDEXED-NEXT:         [  1, _ZTV8Derived1,        250 ] (25.00%)
// COMMON: Instrumentation level: IR  entry_first = 0
// COMMON-NEXT: Functions shown: 1
// COMMON-NEXT: Total functions: 7
// COMMON-NEXT: Maximum function count: 1000
// COMMON-NEXT: Maximum internal block count: 1000
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
// ICTEXT: 470088714870327456
// ICTEXT: # Num Counters:
// ICTEXT: 4
// ICTEXT: # Counter Values:
// ICTEXT: 1000
// ICTEXT: 1000
// ICTEXT: 200
// ICTEXT: 1
// ICTEXT: # Num Value Kinds:
// ICTEXT: 2
// ICTEXT: # ValueKind = IPVK_IndirectCallTarget:
// ICTEXT: 0
// ICTEXT: # NumValueSites:
// ICTEXT: 2
// ICTEXT: 2
// ICTEXT: {{.*}}lib.cpp;_ZN12_GLOBAL__N_18Derived24funcEii:150
// ICTEXT: _ZN8Derived14funcEii:50
// ICTEXT: 2
// ICTEXT: {{.*}}lib.cpp;_ZN12_GLOBAL__N_18Derived2D0Ev:750
// ICTEXT: _ZN8Derived1D0Ev:250
// ICTEXT: # ValueKind = IPVK_VTableTarget:
// ICTEXT: 2
// ICTEXT: # NumValueSites:
// ICTEXT: 2
// ICTEXT: 2
// ICTEXT: {{.*}}lib.cpp;_ZTVN12_GLOBAL__N_18Derived2E:150
// ICTEXT: _ZTV8Derived1:50
// ICTEXT: 2
// ICTEXT: {{.*}}lib.cpp;_ZTVN12_GLOBAL__N_18Derived2E:750
// ICTEXT: _ZTV8Derived1:250

// When vtable value profiles exist, pgo-instr-use pass should annotate them
// even if `-enable-vtable-value-profiling` is not explicitly on.
// RUN: %clangxx -m64 -fprofile-use=test.profdata -fuse-ld=lld -O2 \
// RUN:   -mllvm -print-after=pgo-instr-use -mllvm -filter-print-funcs=main \
// RUN:   -mllvm -print-module-scope lib.cpp main.cpp 2>&1 | FileCheck %s --check-prefix=ANNOTATE

// ANNOTATE-NOT: Inconsistent number of value sites
// ANNOTATE: !{!"VP", i32 2

// When vtable value profiles exist, pgo-instr-use pass will not annotate them
// if `-icp-max-num-vtables` is set to zero.
// RUN: %clangxx -m64 -fprofile-use=test.profdata -fuse-ld=lld -O2 \
// RUN:   -mllvm -icp-max-num-vtables=0 -mllvm -print-after=pgo-instr-use \
// RUN:   -mllvm -filter-print-funcs=main -mllvm -print-module-scope lib.cpp main.cpp 2>&1 | \
// RUN:   FileCheck %s --check-prefix=OMIT

// OMIT: Inconsistent number of value sites
// OMIT-NOT: !{!"VP", i32 2

// Test indirect call promotion transformation using vtable profiles.
// - Build with `-g` to enable debug information.
// - In real world settings, ICP pass is disabled in prelink pipeline. In
//   the postlink pipeline, ICP is enabled after whole-program-devirtualization
//   pass. Do the same thing in this test.
// - Enable `-fwhole-program-vtables` generate type metadata and intrinsics.
// - Enable `-fno-split-lto-unit` and `-Wl,-lto-whole-program-visibility` to
//   preserve type intrinsics for ICP pass.
// RUN: %clangxx -m64  -fprofile-use=test.profdata -Wl,--lto-whole-program-visibility \
// RUN:    -mllvm -disable-icp=true -Wl,-mllvm,-disable-icp=false -fuse-ld=lld \
// RUN:    -g -flto=thin -fwhole-program-vtables -fno-split-lto-unit -O2 \
// RUN:    -mllvm -enable-vtable-value-profiling -Wl,-mllvm,-enable-vtable-value-profiling \
// RUN:    -mllvm -enable-vtable-profile-use \
// RUN:    -Wl,-mllvm,-enable-vtable-profile-use -Rpass=pgo-icall-prom \
// RUN:    -Wl,-mllvm,-print-after=pgo-icall-prom \
// RUN:    -Wl,-mllvm,-filter-print-funcs=main lib.cpp main.cpp 2>&1 \
// RUN:    | FileCheck %s --check-prefixes=REMARK,IR --implicit-check-not="!VP"

// For the indirect call site `ptr->func`
// REMARK: main.cpp:10:19: Promote indirect call to _ZN12_GLOBAL__N_18Derived24funcEii.llvm.{{.*}} with count 150 out of 200, sink 1 instruction(s) and compare 1 vtable(s): {_ZTVN12_GLOBAL__N_18Derived2E.llvm.{{.*}}}
// REMARK: main.cpp:10:19: Promote indirect call to _ZN8Derived14funcEii with count 50 out of 50, sink 1 instruction(s) and compare 1 vtable(s): {_ZTV8Derived1}
//
// For the indirect call site `delete ptr`
// REMARK: main.cpp:12:5: Promote indirect call to _ZN12_GLOBAL__N_18Derived2D0Ev.llvm.{{.*}} with count 750 out of 1000, sink 2 instruction(s) and compare 1 vtable(s): {_ZTVN12_GLOBAL__N_18Derived2E.llvm.{{.*}}}
// REMARK: main.cpp:12:5: Promote indirect call to _ZN8Derived1D0Ev with count 250 out of 250, sink 2 instruction(s) and compare 1 vtable(s): {_ZTV8Derived1}

// The IR matchers for indirect callsite `ptr->func`.
// IR-LABEL: @main
// IR:   [[OBJ:%.*]] = {{.*}}call {{.*}} @_Z10createTypei
// IR:   [[VTABLE:%.*]] = load ptr, ptr [[OBJ]]
// IR:   [[CMP1:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds (i8, ptr @_ZTVN12_GLOBAL__N_18Derived2E.llvm.{{.*}}, i32 16)
// IR:   br i1 [[CMP1]], label %[[BB1:.*]], label %[[BB2:[a-zA-Z0-9_.]+]],
//
// IR: [[BB1]]:
// IR:   [[RESBB1:%.*]] = {{.*}}call {{.*}} @_ZN12_GLOBAL__N_18Derived24funcEii.llvm.{{.*}}
// IR:   br label %[[MERGE0:[a-zA-Z0-9_.]+]]
//
// IR: [[BB2]]:
// IR:   [[CMP2:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds (i8, ptr @_ZTV8Derived1, i32 16)
// IR:   br i1 [[CMP2]], label %[[BB3:.*]], label %[[BB4:[a-zA-Z0-9_.]+]],
//
// IR: [[BB3]]:
// IR:   [[RESBB3:%.*]] = {{.*}}call {{.*}} @_ZN8Derived14funcEii
// IR:   br label %[[MERGE1:[a-zA-Z0-9_.]+]],
//
// IR: [[BB4]]:
// IR:   [[FUNCPTR:%.*]] = load ptr, ptr [[VTABLE]]
// IR:   [[RESBB4:%.*]] = {{.*}}call {{.*}} [[FUNCPTR]]
// IR:   br label %[[MERGE1]]
//
// IR: [[MERGE1]]:
// IR:    [[RES1:%.*]] = phi i32 [ [[RESBB4]], %[[BB4]] ], [ [[RESBB3]], %[[BB3]] ]
// IR:    br label %[[MERGE0]]
//
// IR: [[MERGE0]]:
// IR:    [[RES2:%.*]] = phi i32 [ [[RES1]], %[[MERGE1]] ], [ [[RESBB1]], %[[BB1]] ]

//--- lib.h
#include <stdio.h>
#include <stdlib.h>
class Base {
public:
  virtual int func(int a, int b) = 0;

  virtual ~Base() {};
};

class Derived1 : public Base {
public:
  int func(int a, int b) override;

  ~Derived1() {}
};

__attribute__((noinline)) Base *createType(int a);

//--- lib.cpp
#include "lib.h"

namespace {
class Derived2 : public Base {
public:
  int func(int a, int b) override { return a * (a - b); }

  ~Derived2() {}
};
} // namespace

int Derived1::func(int a, int b) { return a * b; }

Base *createType(int a) {
  Base *base = nullptr;
  if (a % 4 == 0)
    base = new Derived1();
  else
    base = new Derived2();
  return base;
}

//--- main.cpp
#include "lib.h"

int main(int argc, char **argv) {
  int sum = 0;
  for (int i = 0; i < 1000; i++) {
    int a = rand();
    int b = rand();
    Base *ptr = createType(i);
    if (i % 5 == 0)
      sum += ptr->func(b, a);

    delete ptr;
  }
  printf("sum is %d\n", sum);
  return 0;
}
