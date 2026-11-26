// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// MSVC marks this as xfail because it doesn't generate the metadata to display the variable's location in source.
// XFAIL: msvc

int main() {
  int x;
  {
    int x;
    delete &x;
    // CHECK: {{.*}}) 'x' (line [[@LINE-2]])
  }
}

