// Test without serialization:
// RUN: %clang_cc1 -std=c23 -ast-dump %s -ast-dump-filter Test \
// RUN: | FileCheck --strict-whitespace --match-full-lines %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=c23 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple x86_64-unknown-unknown -Wno-unused-value -std=c23 \
// RUN:           -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace --match-full-lines %s


// CHECK:  |   |-value: AddrLabelDiff &&l2 - &&l1
int Test(void) {
  constexpr char ar = &&l2 - &&l1;
l1:
  return 10;
l2:
  return 11;
}


