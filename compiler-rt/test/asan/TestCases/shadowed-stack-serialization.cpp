// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

int main() {
  int x;
  {
    int x;
    delete &x;
    // CHECK: {{.*}}) 'x' (line [[@LINE-2]])
  }
}

