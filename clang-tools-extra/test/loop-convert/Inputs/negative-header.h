#ifndef _CLANG_TOOLS_EXTRA_H_
#define _CLANG_TOOLS_EXTRA_H_

// Single FileCheck line to make sure that no loops are converted.
// CHECK-NOT: for ({{.*[^:]:[^:].*}})
static void loopInHeader() {
  const int N = 10;
  int arr[N];
  int sum = 0;
  for (int i = 0; i < N; ++i)
    sum += arr[i];
}

#endif //_CLANG_TOOLS_EXTRA_H_
