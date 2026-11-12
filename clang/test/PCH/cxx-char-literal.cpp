// RUN: %clang_cc1 -emit-pch -std=c++1z -o %t %s
// RUN: %clang_cc1 -std=c++1z -x ast -ast-print %t | FileCheck %s

// Ensure that character literals are properly surfaced through PCH.

char a = '0';
// CHECK: char a = '0';

char b = L'1';
// CHECK: char b = L'1';

char c = u8'2';
// CHECK: char c = u8'2';

char d = U'3';
// CHECK: char d = U'3';

char e = u'4';
// CHECK: char e = u'4';
