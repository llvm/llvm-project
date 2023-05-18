// RUN: touch %t.hmap

// RUN: %clang_cc1 -x objective-c -emit-pch -o %t.h.pch %S/headermap.h
// RUN: %clang_cc1 -include-pch %t.h.pch %s

// RUN: %clang_cc1 -x objective-c -emit-pch -o %t.h.pch %S/headermap.h
// RUN: %clang_cc1 -include-pch %t.h.pch -I%t.hmap %s

// RUN: %clang_cc1 -x objective-c -I%t.hmap -emit-pch -o %t.h.pch %S/headermap.h
// RUN: %clang_cc1 -include-pch %t.h.pch %s

// RUN: %clang_cc1 -x objective-c -I%t.hmap -emit-pch -o %t.h.pch %S/headermap.h
// RUN: %clang_cc1 -include-pch %t.h.pch -I%t.hmap %s
#import "headermap.h"

