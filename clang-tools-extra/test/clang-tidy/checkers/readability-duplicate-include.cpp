// RUN: %check_clang_tidy %s readability-duplicate-include %t -- -- -isystem %S/Inputs/readability-duplicate-include/system -I %S/Inputs/readability-duplicate-include

int a;
#include <string.h>
int b;
#include <string.h>
int c;
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: duplicate include [readability-duplicate-include]
// CHECK-FIXES:      {{^int a;$}}
// CHECK-FIXES-NEXT: {{^#include <string.h>$}}
// CHECK-FIXES-NEXT: {{^int b;$}}
// CHECK-FIXES-NEXT: {{^int c;$}}

int d;
#include <iostream>
int e;
#include <iostream> // extra stuff that will also be removed
int f;
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: duplicate include
// CHECK-FIXES:      {{^int d;$}}
// CHECK-FIXES-NEXT: {{^#include <iostream>$}}
// CHECK-FIXES-NEXT: {{^int e;$}}
// CHECK-FIXES-NEXT: {{^int f;$}}

int g;
#include "readability-duplicate-include.h"
int h;
#include "readability-duplicate-include.h"
int i;
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: duplicate include
// CHECK-FIXES:      {{^int g;$}}
// CHECK-FIXES-NEXT: {{^#include "readability-duplicate-include.h"$}}
// CHECK-FIXES-NEXT: {{^int h;$}}
// CHECK-FIXES-NEXT: {{^int i;$}}

#include <types.h>

int j;
#include <sys/types.h>
int k;
#include <sys/types.h>
int l;
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: duplicate include
// CHECK-FIXES:      {{^int j;$}}
// CHECK-FIXES-NEXT: {{^#include <sys/types.h>$}}
// CHECK-FIXES-NEXT: {{^int k;$}}
// CHECK-FIXES-NEXT: {{^int l;$}}

int m;
        #          include             <string.h>  // lots of space
int n;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: duplicate include
// CHECK-FIXES:      {{^int m;$}}
// CHECK-FIXES-NEXT: {{^int n;$}}

// defining a macro in the main file resets the included file cache
#define ARBITRARY_MACRO
int o;
#include <sys/types.h>
int p;
// CHECK-FIXES:      {{^int o;$}}
// CHECK-FIXES-NEXT: {{^#include <sys/types.h>$}}
// CHECK-FIXES-NEXT: {{^int p;$}}

// undefining a macro resets the cache
#undef ARBITRARY_MACRO
int q;
#include <sys/types.h>
int r;
// CHECK-FIXES:      {{^int q;$}}
// CHECK-FIXES-NEXT: {{^#include <sys/types.h>$}}
// CHECK-FIXES-NEXT: {{^int r;$}}
