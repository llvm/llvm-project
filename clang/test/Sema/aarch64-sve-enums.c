// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 %s -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns  -fsyntax-only -verify
// expected-no-diagnostics

// This test makes sure that the enum declarations in section "5. Enum
// declarations" of the SVE ACLE [1] are not presented as typedefs in
// `arm_sve.h`. It does so by creating a typedef'd struct with the
// same identifier as the one defined in `arm_sve.h`, then checking that
// it does not overload the enum defined in `arm_sve.h`.
//
// [1] https://developer.arm.com/documentation/100987/latest version 00bet6

typedef struct { float f; } svpattern;
typedef struct { float f; } svprfop;
#include <arm_sve.h>
enum svpattern a1 = SV_ALL;
svpattern b1 = {1.0f};
enum svprfop a2 = SV_PLDL1KEEP;
svprfop b2 = {1.0f};
