// RUN: %clang_cc1 %s -triple s390x-ibm-zos -fexec-charset IBM-1047 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple s390x-ibm-zos -fexec-charset UTF-8 -emit-llvm -o - | FileCheck %s
// REQUIRES: systemz-registered-target

// The mapped name of '#pragma map' is parsed like an asm label: numeric
// escape sequences are IBM-1047 code points, regardless of -fexec-charset.
// '@' is 0x7C (octal 174) in IBM-1047.

void f0(void);
void f1(void);
void f2(void);

// Plain '@' character.
#pragma map(f0, "@F0")
// Octal escape sequence for '@' in IBM-1047.
#pragma map(f1, "\174F1")
// Hex escape sequence for '@' in IBM-1047 ('G' is not a hex digit).
#pragma map(f2, "\x7cG2")

void f0(void) {}
void f1(void) {}
void f2(void) {}

// The mapped name is the symbol name, and the source name is kept in the
// "zos-ppa1-name" function attribute.
// CHECK:      define void @"@F0"() #[[A0:[0-9]+]]
// CHECK:      define void @"@F1"() #[[A1:[0-9]+]]
// CHECK:      define void @"@G2"() #[[A2:[0-9]+]]
// CHECK-DAG:  attributes #[[A0]] = {{{.*}}"zos-ppa1-name"="f0"{{.*}}}
// CHECK-DAG:  attributes #[[A1]] = {{{.*}}"zos-ppa1-name"="f1"{{.*}}}
// CHECK-DAG:  attributes #[[A2]] = {{{.*}}"zos-ppa1-name"="f2"{{.*}}}
