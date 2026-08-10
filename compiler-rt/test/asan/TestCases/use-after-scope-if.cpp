// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

int *p;
bool b = true;

int main() {
  if (b) {
    int x[5];
    p = x+1;
  }
  return *p;  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK:  #0 0x{{.*}} in main {{.*}}.cpp:[[@LINE-2]]
}
