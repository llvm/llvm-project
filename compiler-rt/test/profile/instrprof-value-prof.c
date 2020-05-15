// RUN: %clang_profgen -mllvm -vp-static-alloc=false  -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t-2.profraw %run %t DO_NOT_INSTRUMENT
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata merge -o %t-2.profdata %t-2.profraw
// RUN: llvm-profdata merge -o %t-merged.profdata %t.profraw %t-2.profdata
// RUN: llvm-profdata show --all-functions -ic-targets  %t-2.profdata | FileCheck  %s -check-prefix=NO-VALUE
// RUN: llvm-profdata show --all-functions -ic-targets  %t.profdata | FileCheck  %s
// RUN: llvm-profdata show --all-functions -ic-targets  %t-merged.profdata | FileCheck  %s
//
// RUN: env LLVM_PROFILE_FILE=%t-3.profraw LLVM_VP_BUFFER_SIZE=1 %run %t
// RUN: env LLVM_PROFILE_FILE=%t-4.profraw LLVM_VP_BUFFER_SIZE=8 %run %t
// RUN: env LLVM_PROFILE_FILE=%t-5.profraw LLVM_VP_BUFFER_SIZE=128 %run %t
// RUN: env LLVM_PROFILE_FILE=%t-6.profraw LLVM_VP_BUFFER_SIZE=1024 %run %t
// RUN: env LLVM_PROFILE_FILE=%t-7.profraw LLVM_VP_BUFFER_SIZE=102400 %run %t
// RUN: llvm-profdata merge -o %t-3.profdata %t-3.profraw
// RUN: llvm-profdata merge -o %t-4.profdata %t-4.profraw
// RUN: llvm-profdata merge -o %t-5.profdata %t-5.profraw
// RUN: llvm-profdata merge -o %t-6.profdata %t-6.profraw
// RUN: llvm-profdata merge -o %t-7.profdata %t-7.profraw
// RUN: llvm-profdata show --all-functions -ic-targets  %t-3.profdata | FileCheck  %s
// RUN: llvm-profdata show --all-functions -ic-targets  %t-4.profdata | FileCheck  %s
// RUN: llvm-profdata show --all-functions -ic-targets  %t-5.profdata | FileCheck  %s
// RUN: llvm-profdata show --all-functions -ic-targets  %t-6.profdata | FileCheck  %s
// RUN: llvm-profdata show --all-functions -ic-targets  %t-7.profdata | FileCheck  %s

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

#define DEF_FUNC(x)                                                            \
  void x() {}
#define DEF_2_FUNCS(x) DEF_FUNC(x##_1) DEF_FUNC(x##_2)
#define DEF_4_FUNCS(x) DEF_2_FUNCS(x##_1) DEF_2_FUNCS(x##_2)
#define DEF_8_FUNCS(x) DEF_4_FUNCS(x##_1) DEF_4_FUNCS(x##_2)
#define DEF_16_FUNCS(x) DEF_8_FUNCS(x##_1) DEF_8_FUNCS(x##_2)
#define DEF_32_FUNCS(x) DEF_16_FUNCS(x##_1) DEF_16_FUNCS(x##_2)
#define DEF_64_FUNCS(x) DEF_32_FUNCS(x##_1) DEF_32_FUNCS(x##_2)
#define DEF_128_FUNCS(x) DEF_64_FUNCS(x##_1) DEF_64_FUNCS(x##_2)

#define FUNC_ADDR(x) &x,
#define FUNC_2_ADDRS(x) FUNC_ADDR(x##_1) FUNC_ADDR(x##_2)
#define FUNC_4_ADDRS(x) FUNC_2_ADDRS(x##_1) FUNC_2_ADDRS(x##_2)
#define FUNC_8_ADDRS(x) FUNC_4_ADDRS(x##_1) FUNC_4_ADDRS(x##_2)
#define FUNC_16_ADDRS(x) FUNC_8_ADDRS(x##_1) FUNC_8_ADDRS(x##_2)
#define FUNC_32_ADDRS(x) FUNC_16_ADDRS(x##_1) FUNC_16_ADDRS(x##_2)
#define FUNC_64_ADDRS(x) FUNC_32_ADDRS(x##_1) FUNC_32_ADDRS(x##_2)
#define FUNC_128_ADDRS(x) FUNC_64_ADDRS(x##_1) FUNC_64_ADDRS(x##_2)

DEF_8_FUNCS(callee)
DEF_128_FUNCS(caller)

void *CallerAddrs[] = {FUNC_128_ADDRS(caller)};
void *CalleeAddrs[] = {FUNC_8_ADDRS(callee)};
typedef struct CallerInfo {
    void *CallerAddr;
    uint32_t NS; /* Number value sites. */
} CallerInfo;

CallerInfo CallerInfos[128];

int cmpaddr(const void *p1, const void *p2) {
  CallerInfo *addr1 = (CallerInfo *)p1;
  CallerInfo *addr2 = (CallerInfo *)p2;
  return (intptr_t)addr2->CallerAddr - (intptr_t)addr1->CallerAddr;
}

int main(int argc, const char *argv[]) {
  unsigned S, NS = 0, I, V, doInstrument = 1;
  const __llvm_profile_data *Data, *DataEnd;

  if (argc >= 2 && !strcmp(argv[1], "DO_NOT_INSTRUMENT"))
    doInstrument = 0;

  for (I = 0; I < 128; I++) {
     CallerInfos[I].CallerAddr = CallerAddrs[I];
     CallerInfos[I].NS = I;
  }
  qsort(CallerInfos, sizeof(CallerInfos) / sizeof(CallerInfo), sizeof(CallerInfo),
        cmpaddr);

  /* We will synthesis value profile data for 128 callers functions declared.
   * The number of value sites for each caller function is recorded in
   * the NS field of the CallerInfo object. For each value site, the number of
   * callee values is determined by the site index (modulo 8). The frequency
   * of each callee target synthesized is equal to V + 1, in which V is the
   * index of the target value for the callsite. */

  Data = __llvm_profile_begin_data();
  DataEnd = __llvm_profile_end_data();

  for (; Data < DataEnd; Data = __llvm_profile_iterate_data(Data)) {
    void *func = __llvm_get_function_addr(Data);
    CallerInfo Key, *Res;
    Key.CallerAddr = func;
    Res = (CallerInfo *) bsearch(&Key, CallerInfos, sizeof(CallerInfos) / sizeof(CallerInfo),
                                 sizeof(CallerInfo), cmpaddr);
    if (Res) {
      NS = Res->NS;
      __llvm_profile_set_num_value_sites((__llvm_profile_data *)Data,
                                         0 /*IPVK_IndirectCallTarget */, NS);
      if (!doInstrument) {
        continue;
      }
      for (S = 0; S < NS; S++) {
        for (V = 0; V < S % 8; V++) {
          unsigned C;
          for (C = 0; C < V + 1; C++)
            __llvm_profile_instrument_target((uint64_t)CalleeAddrs[V],
                                             (void *)Data, S);
        }
      }
    }
  }
}

// NO-VALUE: Indirect Call Site Count: 127
// NO-VALUE-NEXT: Indirect Target Results:
// CHECK-LABEL: caller_1_1_1_1_2_2_1:
// CHECK: Indirect Call Site Count: 6
// CHECK-NEXT: Indirect Target Results:
// CHECK-NEXT: [ 1, callee_1_1_1, 1 ]
// CHECK-NEXT: [ 2, callee_1_1_2, 2 ]
// CHECK-NEXT: [ 2, callee_1_1_1, 1 ]
// CHECK-NEXT: [ 3, callee_1_2_1, 3 ]
// CHECK-NEXT: [ 3, callee_1_1_2, 2 ]
// CHECK-NEXT: [ 3, callee_1_1_1, 1 ]
// CHECK-NEXT: [ 4, callee_1_2_2, 4 ]
// CHECK-NEXT: [ 4, callee_1_2_1, 3 ]
// CHECK-NEXT: [ 4, callee_1_1_2, 2 ]
// CHECK-NEXT: [ 4, callee_1_1_1, 1 ]
// CHECK-NEXT: [ 5, callee_2_1_1, 5 ]
// CHECK-NEXT: [ 5, callee_1_2_2, 4 ]
// CHECK-NEXT: [ 5, callee_1_2_1, 3 ]
// CHECK-NEXT: [ 5, callee_1_1_2, 2 ]
// CHECK-NEXT: [ 5, callee_1_1_1, 1 ]
// CHECK-LABEL: caller_2_2_2_2_2_2_2:
// CHECK: Indirect Call Site Count: 127
// CHECK-NEXT: Indirect Target Results:
// CHECK-NEXT:  [ 1, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 2, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 2, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 3, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 3, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 3, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 4, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 4, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 4, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 4, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 5, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 5, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 5, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 5, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 5, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 6, callee_2_1_2, 6 ]
// CHECK-NEXT:  [ 6, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 6, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 6, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 6, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 6, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 7, callee_2_2_1, 7 ]
// CHECK-NEXT:  [ 7, callee_2_1_2, 6 ]
// CHECK-NEXT:  [ 7, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 7, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 7, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 7, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 7, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 9, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 10, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 10, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 11, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 11, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 11, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 12, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 12, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 12, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 12, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 13, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 13, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 13, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 13, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 13, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 14, callee_2_1_2, 6 ]
// CHECK-NEXT:  [ 14, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 14, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 14, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 14, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 14, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 15, callee_2_2_1, 7 ]
// CHECK-NEXT:  [ 15, callee_2_1_2, 6 ]
// CHECK-NEXT:  [ 15, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 15, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 15, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 15, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 15, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 17, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 18, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 18, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 19, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 19, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 19, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 20, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 20, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 20, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 20, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 21, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 21, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 21, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 21, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 21, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 22, callee_2_1_2, 6 ]
// CHECK-NEXT:  [ 22, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 22, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 22, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 22, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 22, callee_1_1_1, 1 ]

