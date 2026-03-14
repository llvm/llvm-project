// REQUIRES: case-insensitive-filesystem

// RUN: rm -rf %t && split-file %s %t
// RUN: sed "s|DIR|%{/t:real}|g" %t/tu.c.in > %t/tu.c
// RUN: %clang_cc1 -fsyntax-only %t/tu.c 2>&1 | FileCheck %s -DDIR=%{/t:real}

//--- header.h
//--- tu.c.in
#import "DIR/Header.h"
// CHECK:      tu.c:1:9: warning: non-portable path to file '"[[DIR]]/header.h"'; specified path differs in case from file name on disk [-Wnonportable-include-path]
// CHECK-NEXT:    1 | #import "[[DIR]]/Header.h"
// CHECK-NEXT:      |         ^~~~~~~~~~~~~~~~~~
// CHECK-NEXT:      |         "[[DIR]]/header.h"
