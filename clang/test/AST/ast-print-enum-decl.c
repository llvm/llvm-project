// First check compiling and printing of this file.
//
// RUN: %clang_cc1 -verify -ast-print %s > %t.c
// RUN: FileCheck --check-prefixes=CHECK,PRINT %s --input-file %t.c
//
// Now check compiling and printing of the printed file.
//
// RUN: echo "// expected""-warning@* 6 {{'T' is deprecated}}" >> %t.c
// RUN: echo "// expected""-note@* 6 {{'T' has been explicitly marked deprecated here}}" >> %t.c
//
// RUN: %clang_cc1 -verify -ast-print %t.c \
// RUN: | FileCheck --check-prefixes=CHECK,PRINT %s

// END.

// CHECK-LABEL: defFirst
void defFirst() {
  // PRINT-NEXT: enum
  // PRINT-DAG:  __attribute__((aligned(16)))
  // PRINT-DAG:  __attribute__((deprecated("")))
  // PRINT-SAME: T {
  // PRINT-NEXT:   E0,
  // PRINT-NEXT:   E1
  // PRINT-NEXT: } *p0;
  // expected-warning@+2 {{'T' is deprecated}}
  // expected-note@+1 2 {{'T' has been explicitly marked deprecated here}}
  enum __attribute__((aligned(16))) __attribute__((deprecated(""))) T {
    E0, E1
  } *p0;

  // PRINT-NEXT: enum T *p1;
  enum T *p1; // expected-warning {{'T' is deprecated}}
}

// CHECK-LABEL: defLast
void defLast() {
  // PRINT-NEXT: enum __attribute__((aligned(16))) T *p0;
  enum __attribute__((aligned(16))) T *p0;

  // PRINT-NEXT: enum __attribute__((deprecated(""))) T {
  // PRINT-NEXT:   E0,
  // PRINT-NEXT:   E1
  // PRINT-NEXT: } *p1;
  // expected-warning@+2 {{'T' is deprecated}}
  // expected-note@+1 {{'T' has been explicitly marked deprecated here}}
  enum __attribute__((deprecated(""))) T { E0, E1 } *p1;
}

// CHECK-LABEL: defMiddle
void defMiddle() {
  // PRINT-NEXT: enum __attribute__((deprecated(""))) T *p0;
  // expected-warning@+2 {{'T' is deprecated}}
  // expected-note@+1 3 {{'T' has been explicitly marked deprecated here}}
  enum __attribute__((deprecated(""))) T *p0;

  // PRINT-NEXT: enum __attribute__((aligned(16))) T {
  // PRINT-NEXT:   E0
  // PRINT-NEXT:   E1
  // PRINT-NEXT: } *p1;
  enum __attribute__((aligned(16))) T { E0, E1 } *p1; // expected-warning {{'T' is deprecated}}

  // PRINT-NEXT: enum T *p2;
  enum T *p2; // expected-warning {{'T' is deprecated}}
}

// CHECK-LABEL: declsOnly
void declsOnly() {
  // FIXME: For some reason, attributes are ignored if they're not on the first
  // declaration and not on the definition.

  // PRINT-NEXT: enum __attribute__((aligned)) T *p0;
  enum __attribute__((aligned)) T *p0;

  // PRINT-NEXT: enum T *p1;
  enum __attribute__((may_alias)) T *p1;

  // PRINT-NEXT: enum T *p2;
  enum T *p2;

  // PRINT-NEXT: enum T *p3;
  enum __attribute__((deprecated(""))) T *p3;

  // PRINT-NEXT: enum T *p4;
  enum T *p4;
}

// Check that tag decl groups stay together in decl contexts.

// PRINT-LABEL: enum DeclGroupAtFileScope {
// PRINT-NEXT:    DeclGroupAtFileScope0
// PRINT-NEXT:  } *DeclGroupAtFileScopePtr;
enum DeclGroupAtFileScope { DeclGroupAtFileScope0 } *DeclGroupAtFileScopePtr;

// PRINT-LABEL: struct DeclGroupInMemberList
struct DeclGroupInMemberList {
  // PRINT-NEXT:  enum T1 {
  // PRINT-NEXT:    T10
  // PRINT-NEXT:  } *p0;
  enum T1 { T10 } *p0;
  // PRINT-NEXT:  enum T2 {
  // PRINT-NEXT:    T20
  // PRINT-NEXT:  } *p1, *p2;
  enum T2 { T20 } *p1, *p2;
  // PRINT-NEXT: };
};

enum fixedEnum : int { fixedEnumerator };
// PRINT-LABEL: enum fixedEnum : int {
// PRINT-NEXT: fixedEnumerator
// PRINT-NEXT: };
