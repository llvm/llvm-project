#ifndef LINKAGE_VALUES_H
#define LINKAGE_VALUES_H

extern "C" {

#define HEADER_SEEK_CUR (1)
#define HEADER_SEEK_END (2)
#define HEADER_SEEK_SET (0)
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'HEADER_SEEK_CUR' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'HEADER_SEEK_END' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'HEADER_SEEK_SET' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: HEADER_SEEK_CUR = (1),
// CHECK-FIXES-NEXT: HEADER_SEEK_END = (2),
// CHECK-FIXES-NEXT: HEADER_SEEK_SET = (0)
// CHECK-FIXES-NEXT: };

}

#endif
