// REQUIRES: systemz-registered-target
// RUN: not --crash %clang_cc1 -triple s390x-ibm-linux -S -O2 -o /dev/null %s 2>&1 | FileCheck %s

// CHECK: error: TRANSLATE length must be a compile-time constant between 1 and 256
// CHECK: error: TRANSLATE length must be a compile-time constant between 1 and 256

void tr_invalid_len_zero(char *src, const unsigned char *table) {
    __builtin_s390_tr(src, 0, table);
}

void tr_invalid_len_260(char *src, const unsigned char *table) {
    __builtin_s390_tr(src, 260, table);
}
