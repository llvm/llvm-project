// Test serialization of late-parsed bounds-safety attributes via Modules
// This verifies that LateParsedAttrType is transformed to CountAttributedType
// before serialization and remains as CountAttributedType after deserialization.

// RUN: rm -rf %t
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fmodules -fmodules-cache-path=%t -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fmodules -fmodules-cache-path=%t -ast-dump-all %s | FileCheck %s
// expected-no-diagnostics

#pragma clang module build bounds_safety_late_parsed
module bounds_safety_late_parsed {}
#pragma clang module contents
#pragma clang module begin bounds_safety_late_parsed

// Test where counted_by references a field declared later
struct LateRefPointer {
  int *__attribute__((counted_by(count))) buf;
  int count;
};

// Test with sized_by referencing later field
struct LateRefSized {
  int *__attribute__((sized_by(size))) data;
  int size;
};

// Test with counted_by_or_null referencing later field
struct LateRefCountedByOrNull {
  int *__attribute__((counted_by_or_null(count))) buf;
  int count;
};

// Test with sized_by_or_null referencing later field
struct LateRefSizedByOrNull {
  int *__attribute__((sized_by_or_null(size))) data;
  int size;
};

// Test with nested struct
struct LateRefNested {
  struct Inner {
    int value;
  } *__attribute__((counted_by(n))) items;
  int n;
};

// Test with multiple late-parsed attributes
struct MultipleLateRefs {
  int *__attribute__((counted_by(count1))) buf1;
  int *__attribute__((sized_by(count2))) buf2;
  int *__attribute__((counted_by_or_null(count3))) buf3;
  int *__attribute__((sized_by_or_null(count4))) buf4;
  int count1;
  int count2;
  int count3;
  int count4;
};

#pragma clang module end
#pragma clang module endbuild

#pragma clang module import bounds_safety_late_parsed

struct LateRefPointer *p1;
struct LateRefSized *p2;
struct LateRefCountedByOrNull *p3;
struct LateRefSizedByOrNull *p4;
struct LateRefNested *p5;
struct MultipleLateRefs *p6;

// CHECK: RecordDecl {{.*}} imported in bounds_safety_late_parsed <undeserialized declarations> struct LateRefPointer definition
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed buf 'int * __counted_by(count)':'int *'
// CHECK-NEXT: `-FieldDecl {{.*}} imported in bounds_safety_late_parsed referenced count 'int'

// CHECK: RecordDecl {{.*}} imported in bounds_safety_late_parsed <undeserialized declarations> struct LateRefSized definition
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed data 'int * __sized_by(size)':'int *'
// CHECK-NEXT: `-FieldDecl {{.*}} imported in bounds_safety_late_parsed referenced size 'int'

// CHECK: RecordDecl {{.*}} imported in bounds_safety_late_parsed <undeserialized declarations> struct LateRefCountedByOrNull definition
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed buf 'int * __counted_by_or_null(count)':'int *'
// CHECK-NEXT: `-FieldDecl {{.*}} imported in bounds_safety_late_parsed referenced count 'int'

// CHECK: RecordDecl {{.*}} imported in bounds_safety_late_parsed <undeserialized declarations> struct LateRefSizedByOrNull definition
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed data 'int * __sized_by_or_null(size)':'int *'
// CHECK-NEXT: `-FieldDecl {{.*}} imported in bounds_safety_late_parsed referenced size 'int'

// CHECK: RecordDecl {{.*}} imported in bounds_safety_late_parsed <undeserialized declarations> struct LateRefNested definition
// CHECK: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed items 'struct Inner * __counted_by(n)':'struct Inner *'
// CHECK: `-FieldDecl {{.*}} imported in bounds_safety_late_parsed referenced n 'int'

// CHECK: RecordDecl {{.*}} imported in bounds_safety_late_parsed <undeserialized declarations> struct MultipleLateRefs definition
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed buf1 'int * __counted_by(count1)':'int *'
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed buf2 'int * __sized_by(count2)':'int *'
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed buf3 'int * __counted_by_or_null(count3)':'int *'
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed buf4 'int * __sized_by_or_null(count4)':'int *'
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed referenced count1 'int'
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed referenced count2 'int'
// CHECK-NEXT: |-FieldDecl {{.*}} imported in bounds_safety_late_parsed referenced count3 'int'
// CHECK-NEXT: `-FieldDecl {{.*}} imported in bounds_safety_late_parsed referenced count4 'int'

// Verify that LateParsedAttrType does not appear in the AST dump
// CHECK-NOT: LateParsedAttr

// Verify the import and variable declarations
// CHECK: ImportDecl {{.*}} implicit bounds_safety_late_parsed
// CHECK: VarDecl {{.*}} p1 'struct LateRefPointer *'
// CHECK: VarDecl {{.*}} p2 'struct LateRefSized *'
// CHECK: VarDecl {{.*}} p3 'struct LateRefCountedByOrNull *'
// CHECK: VarDecl {{.*}} p4 'struct LateRefSizedByOrNull *'
// CHECK: VarDecl {{.*}} p5 'struct LateRefNested *'
// CHECK: VarDecl {{.*}} p6 'struct MultipleLateRefs *'
