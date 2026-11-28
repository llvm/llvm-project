#ifndef NAMESPACE_VALUES_H
#define NAMESPACE_VALUES_H

#define NAMESPACE_HEADER_VALUE 41
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'NAMESPACE_HEADER_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: NAMESPACE_HEADER_VALUE = 41
// CHECK-FIXES-NEXT: };

#define NAMESPACE_HEADER_ESCAPE 42
// CHECK-FIXES: #define NAMESPACE_HEADER_ESCAPE 42

int header_value() {
  return NAMESPACE_HEADER_VALUE + NAMESPACE_HEADER_ESCAPE;
}

#endif
