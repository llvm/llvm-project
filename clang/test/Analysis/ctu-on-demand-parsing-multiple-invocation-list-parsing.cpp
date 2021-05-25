// RUN: rm -rf %t
// RUN: mkdir -p %t

// RUN: %host_cxx %s -fPIC -shared -o %t/mock_open.so

// RUN: echo "void bar(); void foo() { bar(); bar(); }" > %t/trigger.c
// RUN: echo "void bar() {}" > %t/importee.c
// RUN: echo '[{"directory":"%t", "command":"cc -c %t/importee.c", "file": "%t/importee.c"}]' > %t/compile_commands.json
// RUN: %clang_extdef_map -p %t "%t/importee.c" > %t/externalDefMap.txt

// Add an empty invocation list to make the on-demand parsing fail and load it again.
// RUN: echo '' > %t/invocations.yaml

// RUN: cd %t && \
// RUN: LD_PRELOAD=%t/mock_open.so \
// RUN: %clang_cc1 -fsyntax-only -analyze \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=. \
// RUN:   -analyzer-config ctu-invocation-list=invocations.yaml \
// RUN:   %t/trigger.c | FileCheck %s

// REQUIRES: shell, system-linux

// CHECK: {{Opening file invocations.yaml: 1}}
// CHECK-NOT: {{Opening file invocations.yaml: 2}}

#define _GNU_SOURCE 1
#include <dlfcn.h>
#include <fcntl.h>

#include <cassert>
#include <cstdarg>
#include <iostream>
using namespace std;

extern "C" int open(const char *name, int flag, ...) {
  // Log how many times the invocation list is opened.
  if ("invocations.yaml" == string(name)) {
    static unsigned N = 0;
    cout << "Opening file invocations.yaml: " << ++N << endl;
  }

  // The original open function will be called to open the files.
  using open_t = int (*)(const char *, int, mode_t);
  static open_t o_open = nullptr;
  if (!o_open)
    o_open = reinterpret_cast<open_t>(dlsym(RTLD_NEXT, "open"));
  assert(o_open && "Cannot find function `open'.");

  va_list vl;
  va_start(vl, flag);
  auto mode = va_arg(vl, mode_t);
  va_end(vl);
  return o_open(name, flag, mode);
}
