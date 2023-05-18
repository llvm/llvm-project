class B /* Test 1 */ { // CHECK: class B2 /* Test 1 */ {
};

class D : public B /* Test 1 */ { // CHECK: class D : public B2 /* Test 1 */ {
};

// Test 1.
// RUN: clang-rename -force -qualified-name B -new-name B2 -qualified-name E -new-name E2 %s -- | sed 's,//.*,,' | FileCheck %s
