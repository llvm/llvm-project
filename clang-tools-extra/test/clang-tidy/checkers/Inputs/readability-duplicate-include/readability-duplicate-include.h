#ifndef READABILITY_DUPLICATE_INCLUDE_H
#define READABILITY_DUPLICATE_INCLUDE_H

extern int g;
#include "readability-duplicate-include2.h"
extern int h;
#include "readability-duplicate-include2.h"
extern int i;
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: duplicate include
// CHECK-FIXES:      {{^extern int g;$}}
// CHECK-FIXES-NEXT: {{^#include "readability-duplicate-include2.h"$}}
// CHECK-FIXES-NEXT: {{^extern int h;$}}
// CHECK-FIXES-NEXT: {{^extern int i;$}}

#endif
