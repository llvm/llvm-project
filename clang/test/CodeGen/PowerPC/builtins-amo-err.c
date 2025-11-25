// RUN: not %clang_cc1 -triple powerpc-ibm-aix -target-cpu pwr9 \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=AIX32-ERROR
// RUN: not %clang_cc1 -triple powerpc64-ibm-aix -target-cpu pwr9 \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=FC-ERROR

void test_amo() {
  unsigned int *ptr1, value1;
  // AIX32-ERROR-COUNT-2: error: this builtin is only available on 64-bit targets
  __builtin_amo_lwat(ptr1, value1, 0);
  // FC-ERROR: argument value 9 is outside the valid range [0-4, 6, 8]
  __builtin_amo_lwat(ptr1, value1, 9);

  unsigned long int *ptr2, value2;
  // AIX32-ERROR-COUNT-2: error: this builtin is only available on 64-bit targets
  __builtin_amo_ldat(ptr2, value2, 3);
  // FC-ERROR: error: argument value 26 is outside the valid range [0-4, 6, 8]
  __builtin_amo_ldat(ptr2, value2, 26);

  signed int *ptr3, value3;
  // AIX32-ERROR-COUNT-2: error: this builtin is only available on 64-bit targets
  __builtin_amo_lwat_s(ptr3, value3, 0);
  // FC-ERROR: argument value 2 is outside the valid range [0, 5, 7, 8]
  __builtin_amo_lwat_s(ptr3, value3, 2);

  signed long int *ptr4, value4;
  // AIX32-ERROR-COUNT-2: error: this builtin is only available on 64-bit targets
  __builtin_amo_ldat_s(ptr4, value4, 5);
  // FC-ERROR: error: argument value 6 is outside the valid range [0, 5, 7, 8]
  __builtin_amo_ldat_s(ptr4, value4, 6);
}
