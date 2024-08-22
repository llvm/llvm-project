// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_hwasan -Wl,--build-id -g %s -o %t/symbolize.exe
// RUN: echo '[{"prefix": "'"%S"'/", "link": "http://test.invalid/{file}:{line}"}]' > %t/symbolize.exe.linkify
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/symbolize.exe 2>&1 | hwasan_symbolize --html --symbols %t --index | FileCheck %s
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/symbolize.exe 2>&1 | hwasan_symbolize --html --linkify %t/symbolize.exe.linkify --symbols %t --index | FileCheck --check-prefixes=CHECK,LINKIFY %s
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/symbolize.exe 2>&1 | hwasan_symbolize --symbols %t --index | FileCheck %s

#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>

static volatile char sink;

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  char *volatile x = (char *)malloc(10);
  sink = x[100];
  // LINKIFY: <a href="http://test.invalid/hwasan_symbolize.cpp:[[@LINE-1]]">
  // CHECK: hwasan_symbolize.cpp:[[@LINE-2]]
  // CHECK: Cause: heap-buffer-overflow
  // CHECK: allocated by thread {{.*}} here:
  // LINKIFY: <a href="http://test.invalid/hwasan_symbolize.cpp:[[@LINE-6]]">
  // CHECK: hwasan_symbolize.cpp:[[@LINE-7]]
  return 0;
}
