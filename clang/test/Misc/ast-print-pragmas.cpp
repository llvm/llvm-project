// RUN: %clang_cc1 -ast-print %s -o - | FileCheck %s
// RUN: %clang_cc1 -DMS_EXT -fsyntax-only -fms-extensions %s -triple x86_64-pc-win32 -ast-print | FileCheck %s --check-prefix=MS-EXT

// CHECK: #pragma clang loop vectorize_width(4)
// CHECK-NEXT: #pragma clang loop interleave_count(8){{$}}

void test(int *List, int Length) {
  int i = 0;
#pragma clang loop vectorize_width(4)
#pragma clang loop interleave_count(8)
// CHECK-NEXT: while (i < Length)
  while (i < Length) {
    List[i] = i * 2;
    i++;
  }

// CHECK: #pragma clang loop distribute(disable)
// CHECK-NEXT: #pragma clang loop vectorize(enable)
// CHECK-NEXT: #pragma clang loop interleave(disable)

#pragma clang loop distribute(disable)
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(disable)
// CHECK-NEXT: while (i - 1 < Length)
  while (i - 1 < Length) {
    List[i] = i * 2;
    i++;
  }

// CHECK: #pragma clang loop distribute(enable)
// CHECK-NEXT: #pragma clang loop vectorize(disable)
// CHECK-NEXT: #pragma clang loop interleave(enable)

#pragma clang loop distribute(enable)
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(enable)
// CHECK-NEXT: while (i - 2 < Length)
  while (i - 2 < Length) {
    List[i] = i * 2;
    i++;
  }
}

template <int V, int I>
void test_nontype_template_param(int *List, int Length) {
#pragma clang loop vectorize_width(V) interleave_count(I)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

// CHECK: #pragma clang loop vectorize_width(V)
// CHECK: #pragma clang loop interleave_count(I)

void test_templates(int *List, int Length) {
  test_nontype_template_param<2, 4>(List, Length);
}

#ifdef MS_EXT
#pragma init_seg(compiler)
// MS-EXT: #pragma init_seg (.CRT$XCC){{$}}
// MS-EXT-NEXT: int x = 3 __declspec(thread);
int __declspec(thread) x = 3;
#endif //MS_EXT

