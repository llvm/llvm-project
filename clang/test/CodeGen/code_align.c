// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -x c %s %s -o - | FileCheck -check-prefix=CHECK-C %s
// RUN: %clang_cc1 -fsyntax-only -emit-llvm -x c++ -std=c++11 %s -o - | FileCheck %s --check-prefixes CHECK-C,CHECK-CPP

// CHECK-C: br label %for.cond, !llvm.loop ![[MD_FP:[0-9]+]]
// CHECK-C: br label %while.cond, !llvm.loop ![[MD_FP_1:[0-9]+]]
// CHECK-C: br i1 %cmp3, label %do.body, label %do.end, !llvm.loop ![[MD_FP_2:[0-9]+]]
// CHECK-C: br label %for.cond5, !llvm.loop ![[MD_FP_3:[0-9]+]]

// CHECK-CPP: br label %for.cond, !llvm.loop ![[MD_FP_4:[0-9]+]]
// CHECK-CPP: br label %for.cond2, !llvm.loop ![[MD_FP_5:[0-9]+]]

void bar(int);
void code_align() {
  int a[10];
  // CHECK-C: ![[MD_FP]] = distinct !{![[MD_FP]], ![[MP:[0-9]+]], ![[MD_code_align:[0-9]+]]}
  // CHECK-C-NEXT: ![[MP]] = !{!"llvm.loop.mustprogress"}
  // CHECK-C-NEXT: ![[MD_code_align]] = !{!"llvm.loop.align", i32 4}
  [[clang::code_align(4)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // CHECK-C: ![[MD_FP_1]] = distinct !{![[MD_FP_1]], ![[MP]], ![[MD_code_align_1:[0-9]+]]}
  // CHECK-C-NEXT: ![[MD_code_align_1]] = !{!"llvm.loop.align", i32 16}
  int i = 0;
  [[clang::code_align(16)]] while (i < 60) {
    a[i] += 3;
  }

  // CHECK-C: ![[MD_FP_2]] = distinct !{![[MD_FP_2]], ![[MP]], ![[MD_code_align_2:[0-9]+]]}
  // CHECK-C-NEXT: ![[MD_code_align_2]] = !{!"llvm.loop.align", i32 8}
  int b = 10;
  [[clang::code_align(8)]] do {
    b = b + 1;
  } while (b < 20);

  // CHECK-C: ![[MD_FP_3]] = distinct !{![[MD_FP_3]], ![[MP]], ![[MD_code_align_3:[0-9]+]]}
  // CHECK-C-NEXT: ![[MD_code_align_3]] = !{!"llvm.loop.align", i32 64}
  [[clang::code_align(64)]]
  for(int I=0; I<128; ++I) { bar(I); }
}

#if __cplusplus >= 201103L
template <int A, int B>
void code_align_cpp() {
  int a[10];	
  // CHECK-CPP: ![[MD_FP_4]] = distinct !{![[MD_FP_4]], ![[MP]], ![[MD_code_align_4:[0-9]+]]}
  // CHECK-CPP-NEXT: ![[MD_code_align_4]] = !{!"llvm.loop.align", i32 32}
  [[clang::code_align(A)]] for (int i = 0; i != 10; ++i)
    a[i] = 0;

  // CHECK-CPP: ![[MD_FP_5]] = distinct !{![[MD_FP_5]], ![[MD_code_align]]}
  int c[] = {0, 1, 2, 3, 4, 5};
  [[clang::code_align(B)]] for (int n : c) { n *= 2; }
}

int main() {
  code_align_cpp<32, 4>();
  return 0;
}
#endif
