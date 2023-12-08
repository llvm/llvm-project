#ifndef MODERNIZE_MACRO_TO_ENUM2_H
#define MODERNIZE_MACRO_TO_ENUM2_H

#include "modernize-macro-to-enum3.h"

#define GG2_RED 0xFF0000
#define GG2_GREEN 0x00FF00
#define GG2_BLUE 0x0000FF
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'GG2_RED' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'GG2_GREEN' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'GG2_BLUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: GG2_RED = 0xFF0000,
// CHECK-FIXES-NEXT: GG2_GREEN = 0x00FF00,
// CHECK-FIXES-NEXT: GG2_BLUE = 0x0000FF
// CHECK-FIXES-NEXT: };

#if 1
#define RR2_RED 1
#define RR2_GREEN 2
#define RR2_BLUE 3
#endif

#endif
