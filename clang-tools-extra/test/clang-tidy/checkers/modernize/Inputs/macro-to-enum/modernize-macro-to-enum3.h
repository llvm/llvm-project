#pragma once

#define GG3_RED 0xFF0000
#define GG3_GREEN 0x00FF00
#define GG3_BLUE 0x0000FF
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'GG3_RED' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'GG3_GREEN' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'GG3_BLUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: GG3_RED = 0xFF0000,
// CHECK-FIXES-NEXT: GG3_GREEN = 0x00FF00,
// CHECK-FIXES-NEXT: GG3_BLUE = 0x0000FF
// CHECK-FIXES-NEXT: };

#if 1
#define RR3_RED 1
#define RR3_GREEN 2
#define RR3_BLUE 3
#endif
