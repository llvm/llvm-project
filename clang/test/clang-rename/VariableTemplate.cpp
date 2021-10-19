template <typename T, int U>
bool Foo = true;  // CHECK: bool Bar = true;

// explicit template specialization
template <>
bool Foo<int, 0> = false; // CHECK: bool Bar<int, 0> = false;

// partial template specialization
template <typename T>
bool Foo<T, 1> = false; // bool Bar<x, 1> = false;

void k() {
  // ref to the explicit template specialization
  Foo<int, 0>;   // CHECK: Bar<int, 0>;
  // ref to the primary template.
  Foo<double, 2>;   // CHECK: Bar<double, 2>;
}


// Test 1.
// RUN: clang-rename -offset=34 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=128 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=248 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 4.
// RUN: clang-rename -offset=357 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 5.
// RUN: clang-rename -offset=431 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
