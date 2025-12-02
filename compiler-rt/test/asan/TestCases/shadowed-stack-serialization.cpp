// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %if target={{.*aix.*}} %{ %env_asan_opts=enable_unmalloced_free_check=1 %} not %run %t 2>&1 \
// RUN: | FileCheck %s

int main() {
  int x;
  {
    int x;
    delete &x;
    // CHECK: {{.*}}) 'x' (line [[@LINE-2]])
  }
}

