// RUN: %clangxx_asan -fno-sanitize-address-use-odr-indicator -fPIC %s -o %t
// RUN: %env_asan_opts=report_globals=2 %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,INDICATOR0

// RUN: %clangxx_asan -fsanitize-address-use-odr-indicator -fPIC %s -o %t
// RUN: %env_asan_opts=report_globals=2 %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,INDICATOR1

#include <stdio.h>

int test_global_1;
// INDICATOR0-DAG: Added Global{{.*}} name=test_global_1{{.*}} odr_indicator={{0x0+$}}
// INDICATOR1-DAG: Added Global{{.*}} name=test_global_1{{.*}} odr_indicator={{0x0*[^0]+.*$}}

static int test_global_2;
// CHECK-DAG: Added Global{{.*}} name=test_global_2{{.*}} source={{.*}} module={{.*}}.cpp.tmp {{.*}} odr_indicator={{0xf+$}}

namespace {
static int test_global_3;
// CHECK-DAG: Added Global{{.*}} name={{.*}}::test_global_3{{.*}} source={{.*}} module={{.*}}.cpp.tmp {{.*}} odr_indicator={{0xf+$}}
} // namespace

int main() {
  const char f[] = "%d %d %d\n";
  // CHECK-DAG: Added Global{{.*}} name=__const.main.f{{.*}} source={{.*}} module={{.*}}.cpp.tmp {{.*}} odr_indicator={{0xf+$}}
  printf(f, test_global_1, test_global_2, test_global_3);
  return 0;
}
