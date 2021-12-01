// RUN: %clang_cc1 -triple aarch64_be-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s

// expected-error@* {{Big endian is currently not supported for arm_sve.h}}
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>
