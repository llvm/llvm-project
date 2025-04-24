// RUN: %clang_cc1 -fbounds-safety %s -I %S/include -verify=legacy,legacy-incorrect,common-incorrect,common
// RUN: %clang_cc1 -fbounds-safety %s -I %S/include -verify=strict,common -fno-bounds-safety-relaxed-system-headers
// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=all %s -DTEST_COMPOUND_LITERALS -I %S/include -verify=incorrect,common-incorrect,common
// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=all %s -DTEST_COMPOUND_LITERALS -I %S/include -verify=strict,common -fno-bounds-safety-relaxed-system-headers
#include <bounds-checks-inline.h>
