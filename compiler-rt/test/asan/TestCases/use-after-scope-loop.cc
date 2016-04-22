// RUN: %clangxx_asan -O1 -mllvm -asan-use-after-scope=1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

int *p[3];

int main() {
  for (int i = 0; i < 3; i++) {
    int x;
    p[i] = &x;
  }
  return **p;  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK: #0 0x{{.*}} in main {{.*}}.cc:[[@LINE-2]]
}
