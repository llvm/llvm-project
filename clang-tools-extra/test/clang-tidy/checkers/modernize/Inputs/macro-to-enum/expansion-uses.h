#ifndef EXPANSION_USES_H
#define EXPANSION_USES_H

#define HEADER_SAFE (13)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'HEADER_SAFE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: HEADER_SAFE = (13)
// CHECK-FIXES-NEXT: };

#define HEADER_BAD (14)
// CHECK-FIXES: #define HEADER_BAD (14)

#define HEADER_LATE_BAD (15)
// CHECK-FIXES: #define HEADER_LATE_BAD (15)

void header_uses() {
  use_one(HEADER_SAFE);
  use_one HEADER_BAD;
}

#endif
