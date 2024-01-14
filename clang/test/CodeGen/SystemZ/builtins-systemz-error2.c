// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -triple s390x-ibm-linux -S -emit-llvm %s -verify -o -

__int128 f0(__int128 a, __int128 b) {
  __builtin_tbegin ((void *)0);    // expected-error {{'__builtin_tbegin' needs target feature transactional-execution}}
  return __builtin_s390_vaq(a, b); // expected-error {{'__builtin_s390_vaq' needs target feature vector}}
}

