// Test serialization of late-parsed bounds-safety attributes via PCH
// This verifies that LateParsedAttrType is transformed to CountAttributedType
// before serialization and remains as CountAttributedType after deserialization.

// RUN: %clang_cc1 -fexperimental-late-parse-attributes -include %S/Inputs/bounds-safety-attributed-type-late-parsed.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -emit-pch -o %t %S/Inputs/bounds-safety-attributed-type-late-parsed.h
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -include-pch %t -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -include-pch %t -ast-print %s | FileCheck %s --check-prefix PRINT
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix DUMP
// expected-no-diagnostics

// PRINT: struct LateRefPointer {
// PRINT-NEXT:     int * __counted_by(count)buf;
// PRINT-NEXT:     int count;
// PRINT-NEXT: };

// PRINT: struct LateRefSized {
// PRINT-NEXT:     int * __sized_by(size)data;
// PRINT-NEXT:     int size;
// PRINT-NEXT: };

// PRINT: struct LateRefCountedByOrNull {
// PRINT-NEXT:     int * __counted_by_or_null(count)buf;
// PRINT-NEXT:     int count;
// PRINT-NEXT: };

// PRINT: struct LateRefSizedByOrNull {
// PRINT-NEXT:     int * __sized_by_or_null(size)data;
// PRINT-NEXT:     int size;
// PRINT-NEXT: };

// DUMP: RecordDecl {{.*}} imported <undeserialized declarations> struct LateRefPointer definition
// DUMP-NEXT: |-FieldDecl {{.*}} imported buf 'int * __counted_by(count)':'int *'
// DUMP-NEXT: `-FieldDecl {{.*}} imported referenced count 'int'

// DUMP: RecordDecl {{.*}} imported <undeserialized declarations> struct LateRefSized definition
// DUMP-NEXT: |-FieldDecl {{.*}} imported data 'int * __sized_by(size)':'int *'
// DUMP-NEXT: `-FieldDecl {{.*}} imported referenced size 'int'

// DUMP: RecordDecl {{.*}} imported <undeserialized declarations> struct LateRefCountedByOrNull definition
// DUMP-NEXT: |-FieldDecl {{.*}} imported buf 'int * __counted_by_or_null(count)':'int *'
// DUMP-NEXT: `-FieldDecl {{.*}} imported referenced count 'int'

// DUMP: RecordDecl {{.*}} imported <undeserialized declarations> struct LateRefSizedByOrNull definition
// DUMP-NEXT: |-FieldDecl {{.*}} imported data 'int * __sized_by_or_null(size)':'int *'
// DUMP-NEXT: `-FieldDecl {{.*}} imported referenced size 'int'

// DUMP: RecordDecl {{.*}} imported <undeserialized declarations> struct LateRefNested definition
// DUMP: |-FieldDecl {{.*}} imported items 'struct Inner * __counted_by(n)':'struct Inner *'
// DUMP: `-FieldDecl {{.*}} imported referenced n 'int'

// DUMP: RecordDecl {{.*}} imported <undeserialized declarations> struct MultipleLateRefs definition
// DUMP-NEXT: |-FieldDecl {{.*}} imported buf1 'int * __counted_by(count1)':'int *'
// DUMP-NEXT: |-FieldDecl {{.*}} imported buf2 'int * __sized_by(count2)':'int *'
// DUMP-NEXT: |-FieldDecl {{.*}} imported buf3 'int * __counted_by_or_null(count3)':'int *'
// DUMP-NEXT: |-FieldDecl {{.*}} imported buf4 'int * __sized_by_or_null(count4)':'int *'
// DUMP-NEXT: |-FieldDecl {{.*}} imported referenced count1 'int'
// DUMP-NEXT: |-FieldDecl {{.*}} imported referenced count2 'int'
// DUMP-NEXT: |-FieldDecl {{.*}} imported referenced count3 'int'
// DUMP-NEXT: `-FieldDecl {{.*}} imported referenced count4 'int'

// DUMP: RecordDecl {{.*}} imported <undeserialized declarations> struct LateRefAnon definition
// DUMP-NEXT: |-FieldDecl {{.*}} imported buf 'int * __counted_by(count)':'int *'
// DUMP: `-IndirectFieldDecl {{.*}} imported implicit referenced count 'int'

// Verify that LateParsedAttrType does not appear in the AST dump
// DUMP-NOT: LateParsedAttr
