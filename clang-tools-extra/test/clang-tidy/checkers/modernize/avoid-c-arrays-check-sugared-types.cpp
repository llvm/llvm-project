// RUN: %check_clang_tidy -std=c++17 %s modernize-avoid-c-arrays %t -- \
// RUN:  -config='{CheckOptions: { modernize-avoid-c-arrays.CheckSugaredTypes: true }}'

int a[] = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead

int b[1];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead

void foo() {
  int c[b[0]];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C VLA arrays, use 'std::vector' instead

  using d = decltype(c);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not declare C VLA arrays, use 'std::vector' instead
  // CHECK-MESSAGES: :[[@LINE-2]]:13: note: declared type 'decltype(c)' resolves to 'int[b[0]]'

  d e;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C VLA arrays, use 'std::vector' instead
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: declared type 'd' resolves to 'int[b[0]]'
}

template <typename T, int Size>
class array {
  T d[Size];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead

  int e[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead
};

array<int[4], 2> d;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use 'std::array' instead

using k = int[4];
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: do not declare C-style arrays, use 'std::array' instead

array<k, 2> dk;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use 'std::array' instead
// CHECK-MESSAGES: :[[@LINE-2]]:7: note: declared type 'k' resolves to 'int[4]'

template <typename T>
class unique_ptr {
  T *d;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead
  // CHECK-MESSAGES: :[[@LINE-2]]:3: note: array type 'int[]' here

  int e[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead
};

unique_ptr<int[]> d2;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use 'std::array' instead

using k2 = int[];
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use 'std::array' instead

unique_ptr<k2> dk2;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use 'std::array' instead
// CHECK-MESSAGES: :[[@LINE-2]]:12: note: declared type 'k2' resolves to 'int[]'

// Some header
extern "C" {

int f[] = {1, 2};

int j[1];

inline void bar() {
  {
    int j[j[0]];
  }
}

extern "C++" {
int f3[] = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead

int j3[1];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead

struct Foo {
  int f3[3] = {1, 2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead

  int j3[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead
};
}

struct Bar {

  int f[3] = {1, 2};

  int j[1];
};
}

const char name[] = "Some string";
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

void takeCharArray(const char name[]);
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: do not declare C-style arrays, use 'std::array' or 'std::vector' instead [modernize-avoid-c-arrays]

using MyIntArray = int[10];
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

using MyIntArray2D = MyIntArray[10];
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
// CHECK-MESSAGES: :[[@LINE-2]]:22: note: declared type 'MyIntArray[10]' resolves to 'int[10][10]'

using MyIntArray3D = MyIntArray2D[10];
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
// CHECK-MESSAGES: :[[@LINE-2]]:22: note: declared type 'MyIntArray2D[10]' resolves to 'int[10][10][10]'

MyIntArray3D array3D;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
// CHECK-MESSAGES: :[[@LINE-2]]:1: note: declared type 'MyIntArray3D' resolves to 'int[10][10][10]'

MyIntArray2D array2D;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
// CHECK-MESSAGES: :[[@LINE-2]]:1: note: declared type 'MyIntArray2D' resolves to 'int[10][10]'

MyIntArray array1D;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
// CHECK-MESSAGES: :[[@LINE-2]]:1: note: declared type 'MyIntArray' resolves to 'int[10]'
