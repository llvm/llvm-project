// RUN: %clang_cc1 -include %S/Inputs/bounds-safety-attributed-type.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/Inputs/bounds-safety-attributed-type.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s
// RUN: %clang_cc1 -include-pch %t -ast-print %s | FileCheck %s --check-prefix PRINT
// RUN: %clang_cc1 -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix DUMP
// expected-no-diagnostics

// PRINT: 	   struct Test {
// PRINT-NEXT:   int count;
// PRINT-NEXT:   int fam[] __counted_by(count);
// PRINT-NEXT: };

// DUMP:        RecordDecl {{.*}} imported <undeserialized declarations> struct Test definition
// DUMP-NEXT:   |-FieldDecl {{.*}} imported referenced count 'int'
// DUMP-NEXT:   `-FieldDecl {{.*}} imported fam 'int[] __counted_by(count)':'int[]'
