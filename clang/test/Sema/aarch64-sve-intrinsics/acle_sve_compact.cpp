// RUN: %clang_cc1 -triple aarch64-none-linux-gnu \
// RUN:   -target-feature +sve -target-feature +sme -target-feature +ssve -target-feature +sme2p2 \
// RUN:   -fsyntax-only -verify %s
// REQUIRES: aarch64-registered-target
// expected-no-diagnostics


#include <arm_sve.h>

void test_svcompact_streaming(svbool_t pg, svfloat32_t op) __arm_streaming {
    svcompact(pg, op);
}