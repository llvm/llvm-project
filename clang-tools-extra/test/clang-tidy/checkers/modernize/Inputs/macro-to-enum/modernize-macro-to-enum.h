#if !defined(MODERNIZE_MACRO_TO_ENUM_H)
#define MODERNIZE_MACRO_TO_ENUM_H

#include "modernize-macro-to-enum2.h"

#define GG_RED 0xFF0000
#define GG_GREEN 0x00FF00
#define GG_BLUE 0x0000FF
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'GG_RED' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'GG_GREEN' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'GG_BLUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: GG_RED = 0xFF0000,
// CHECK-FIXES-NEXT: GG_GREEN = 0x00FF00,
// CHECK-FIXES-NEXT: GG_BLUE = 0x0000FF
// CHECK-FIXES-NEXT: };

#if 1
#define RR_RED 1
#define RR_GREEN 2
#define RR_BLUE 3
#endif

#endif
