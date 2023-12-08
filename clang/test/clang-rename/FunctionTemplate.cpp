template <typename T>
void Foo(T t); // CHECK: void Bar(T t);

template <>
void Foo(int a); // CHECK: void Bar(int a);

void test() {
  Foo<double>(1); // CHECK: Bar<double>(1);
}

// Test 1.
// RUN: clang-rename -offset=28 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=81 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=137 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
