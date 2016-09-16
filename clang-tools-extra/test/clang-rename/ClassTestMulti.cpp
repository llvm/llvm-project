class Foo1 /* Offset 1 */ { // CHECK: class Bar1 /* Offset 1 */ {
};

class Foo2 /* Offset 2 */ { // CHECK: class Bar2 /* Offset 2 */ {
};

// Test 1.
// RUN: clang-rename -offset=6 -new-name=Bar1 -offset=76 -new-name=Bar2 %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
