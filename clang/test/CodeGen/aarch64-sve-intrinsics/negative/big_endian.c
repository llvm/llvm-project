// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64_be-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s

// expected-error@* {{Big endian is currently not supported for arm_sve.h}}
#include <arm_sve.h>
