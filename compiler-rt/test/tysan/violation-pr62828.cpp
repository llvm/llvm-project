// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// https://github.com/llvm/llvm-project/issues/62828
#include <stdio.h>

typedef int int_v8[8];
typedef short short_v8[8];
short *test1(int_v8 *cast_c_array, short_v8 *shuf_c_array1, int *ptr) {
  int *input1 = reinterpret_cast<int *>(((int_v8 *)(cast_c_array)));
  short *input2 = reinterpret_cast<short *>(reinterpret_cast<int_v8 *>(input1));

  short *output1 = reinterpret_cast<short *>(((short_v8 *)(shuf_c_array1)));
  short *output2 =
      reinterpret_cast<short *>(reinterpret_cast<short_v8 *>(output1));

  for (int r = 0; r < 8; ++r) {
    int tmp = (int)((r * 4) + ptr[r]);
    if ((ptr[r] / 4) == 0) {
      int *input = reinterpret_cast<int *>(((int_v8 *)(cast_c_array)));
      input[r] = tmp;
    }
  }

  // CHECK:      ERROR: TypeSanitizer: type-aliasing-violation on address
  // CHECK-NEXT: READ of size 2 at {{.+}} with type short accesses an existing object of type int
  // CHECK-NEXT:    in test1(int (*) [8], short (*) [8], int*) {{.*/?}}violation-pr62828.cpp:29
  for (int i3 = 0; i3 < 4; ++i3) {
    output2[i3] = input2[(i3 * 2)];
  }
  return output2;
}

int main() {
  int_v8 in[4] = {{4, 4, 4, 4}};
  short_v8 out[4] = {{0}};
  int ptr[8] = {2};
  test1(in, out, ptr);
  short *p = reinterpret_cast<short *>(out);
  for (int i = 0; i < 32; i++) {
    printf("%d ", p[i]);
  }
  return 0;
}
