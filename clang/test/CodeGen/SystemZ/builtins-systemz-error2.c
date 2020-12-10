// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -triple s390x-ibm-linux -S -emit-llvm %s -verify -o -

typedef __attribute__((vector_size(16))) char v16i8;

v16i8 f0(v16i8 a, v16i8 b) {
  __builtin_tbegin ((void *)0);         // expected-error {{'__builtin_tbegin' needs target feature transactional-execution}}
  v16i8 tmp = __builtin_s390_vaq(a, b); // expected-error {{'__builtin_s390_vaq' needs target feature vector}}
  return tmp;
}

