// RUN: %clang_profgen -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata show --all-functions -ic-targets  %t.profdata > %t.out
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-1 < %t.out
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-2 < %t.out
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-3 < %t.out
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-4 < %t.out
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-5 < %t.out
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-6 < %t.out

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct __llvm_profile_data __llvm_profile_data;
const __llvm_profile_data *__llvm_profile_begin_data(void);
const __llvm_profile_data *__llvm_profile_end_data(void);
void __llvm_profile_set_num_value_sites(__llvm_profile_data *Data,
                                        uint32_t ValueKind,
                                        uint16_t NumValueSites);
__llvm_profile_data *
__llvm_profile_iterate_data(const __llvm_profile_data *Data);
void *__llvm_get_function_addr(const __llvm_profile_data *Data);
void __llvm_profile_instrument_target(uint64_t TargetValue, void *Data,
                                      uint32_t CounterIndex);
void callee1() {}
void callee2() {}

void caller_without_value_site1() {}
void caller_with_value_site_never_called1() {}
void caller_with_vp1() {}
void caller_with_value_site_never_called2() {}
void caller_without_value_site2() {}
void caller_with_vp2() {}

void (*callee1Ptr)();
void (*callee2Ptr)();

void __attribute__ ((noinline)) setFunctionPointers () {
  callee1Ptr = callee1;
  callee2Ptr = callee2;
}

int main(int argc, const char *argv[]) {
  unsigned S, NS = 10, V;
  const __llvm_profile_data *Data, *DataEnd;

  setFunctionPointers();
  Data = __llvm_profile_begin_data();
  DataEnd = __llvm_profile_end_data();
  for (; Data < DataEnd; Data = __llvm_profile_iterate_data(Data)) {
    void *func = __llvm_get_function_addr(Data);
    if (func == caller_without_value_site1 ||
        func == caller_without_value_site2 ||
        func == callee1 || func == callee2 || func == main)
      continue;

    __llvm_profile_set_num_value_sites((__llvm_profile_data *)Data,
                                       0 /*IPVK_IndirectCallTarget */, 10);

    if (func == caller_with_value_site_never_called1 ||
        func == caller_with_value_site_never_called2)
      continue;
    for (S = 0; S < NS; S++) {
      unsigned C;
      for (C = 0; C < S + 1; C++) {
        __llvm_profile_instrument_target((uint64_t)callee1Ptr, (void *)Data, S);
        if (C % 2 == 0)
          __llvm_profile_instrument_target((uint64_t)callee2Ptr, (void *)Data, S);
      }
    }
  }
}

// CHECK-1-LABEL:   caller_with_value_site_never_called2:
// CHECK-1-NEXT:    Hash: 0x0000000000000000
// CHECK-1-NEXT:    Counters:
// CHECK-1-NEXT:    Function count
// CHECK-1-NEXT:    Indirect Call Site Count: 10
// CHECK-1-NEXT:    Indirect Target Results: 
// CHECK-2-LABEL:   caller_with_vp2:
// CHECK-2-NEXT:    Hash: 0x0000000000000000
// CHECK-2-NEXT:    Counters:
// CHECK-2-NEXT:    Function count:
// CHECK-2-NEXT:    Indirect Call Site Count: 10
// CHECK-2-NEXT:    Indirect Target Results: 
// CHECK-2-NEXT:	[ 0, callee1, 1 ]
// CHECK-2-NEXT:	[ 0, callee2, 1 ]
// CHECK-2-NEXT:	[ 1, callee1, 2 ]
// CHECK-2-NEXT:	[ 1, callee2, 1 ]
// CHECK-2-NEXT:	[ 2, callee1, 3 ]
// CHECK-2-NEXT:	[ 2, callee2, 2 ]
// CHECK-2-NEXT:	[ 3, callee1, 4 ]
// CHECK-2-NEXT:	[ 3, callee2, 2 ]
// CHECK-2-NEXT:	[ 4, callee1, 5 ]
// CHECK-2-NEXT:	[ 4, callee2, 3 ]
// CHECK-2-NEXT:	[ 5, callee1, 6 ]
// CHECK-2-NEXT:	[ 5, callee2, 3 ]
// CHECK-2-NEXT:	[ 6, callee1, 7 ]
// CHECK-2-NEXT:	[ 6, callee2, 4 ]
// CHECK-2-NEXT:	[ 7, callee1, 8 ]
// CHECK-2-NEXT:	[ 7, callee2, 4 ]
// CHECK-2-NEXT:	[ 8, callee1, 9 ]
// CHECK-2-NEXT:	[ 8, callee2, 5 ]
// CHECK-2-NEXT:	[ 9, callee1, 10 ]
// CHECK-2-NEXT:	[ 9, callee2, 5 ]
// CHECK-3-LABEL:   caller_with_vp1:
// CHECK-3-NEXT:    Hash: 0x0000000000000000
// CHECK-3-NEXT:    Counters:
// CHECK-3-NEXT:    Function count
// CHECK-3-NEXT:    Indirect Call Site Count: 10
// CHECK-3-NEXT:    Indirect Target Results: 
// CHECK-3-NEXT:	[ 0, callee1, 1 ]
// CHECK-3-NEXT:	[ 0, callee2, 1 ]
// CHECK-3-NEXT:	[ 1, callee1, 2 ]
// CHECK-3-NEXT:	[ 1, callee2, 1 ]
// CHECK-3-NEXT:	[ 2, callee1, 3 ]
// CHECK-3-NEXT:	[ 2, callee2, 2 ]
// CHECK-3-NEXT:	[ 3, callee1, 4 ]
// CHECK-3-NEXT:	[ 3, callee2, 2 ]
// CHECK-3-NEXT:	[ 4, callee1, 5 ]
// CHECK-3-NEXT:	[ 4, callee2, 3 ]
// CHECK-3-NEXT:	[ 5, callee1, 6 ]
// CHECK-3-NEXT:	[ 5, callee2, 3 ]
// CHECK-3-NEXT:	[ 6, callee1, 7 ]
// CHECK-3-NEXT:	[ 6, callee2, 4 ]
// CHECK-3-NEXT:	[ 7, callee1, 8 ]
// CHECK-3-NEXT:	[ 7, callee2, 4 ]
// CHECK-3-NEXT:	[ 8, callee1, 9 ]
// CHECK-3-NEXT:	[ 8, callee2, 5 ]
// CHECK-3-NEXT:	[ 9, callee1, 10 ]
// CHECK-3-NEXT:	[ 9, callee2, 5 ]
// CHECK-4-LABEL:   caller_with_value_site_never_called1:
// CHECK-4-NEXT:    Hash: 0x0000000000000000
// CHECK-4-NEXT:    Counters:
// CHECK-4-NEXT:    Function count:
// CHECK-4-NEXT:    Indirect Call Site Count: 10
// CHECK-4-NEXT:    Indirect Target Results: 
// CHECK-5-LABEL:   caller_without_value_site2:
// CHECK-5-NEXT:    Hash: 0x0000000000000000
// CHECK-5-NEXT:    Counters:
// CHECK-5-NEXT:    Function count:
// CHECK-5-NEXT:    Indirect Call Site Count: 0
// CHECK-5-NEXT:    Indirect Target Results: 
// CHECK-6-LABEL:   caller_without_value_site1:
// CHECK-6-NEXT:    Hash: 0x0000000000000000
// CHECK-6-NEXT:    Counters:
// CHECK-6-NEXT:    Function count:
// CHECK-6-NEXT:    Indirect Call Site Count: 0
// CHECK-6-NEXT:    Indirect Target Results: 
