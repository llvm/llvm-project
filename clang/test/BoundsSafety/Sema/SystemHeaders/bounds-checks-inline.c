// RUN: %clang_cc1 -fbounds-safety %s -I %S/include -verify=incorrect,common-incorrect,common
// RUN: %clang_cc1 -fbounds-safety %s -I %S/include -verify=strict,common -fno-bounds-safety-relaxed-system-headers
#include "bounds-checks-inline.h"
