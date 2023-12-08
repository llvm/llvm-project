// RUN: rm -rf %T/out
// RUN: clang-tidy -enable-check-profile -checks='-*,readability-function-size' -store-check-profile=%T/out %s -- 2>&1 | not FileCheck --match-full-lines -implicit-check-not='{{warning:|error:}}' -check-prefix=CHECK-CONSOLE %s
// RUN: cat %T/out/*-clang-tidy-store-check-profile-one-tu.cpp.json | FileCheck --match-full-lines -implicit-check-not='{{warning:|error:}}' -check-prefix=CHECK-FILE %s
// RUN: rm -rf %T/out
// RUN: clang-tidy -enable-check-profile -checks='-*,readability-function-size' -store-check-profile=%T/out %s -- 2>&1
// RUN: cat %T/out/*-clang-tidy-store-check-profile-one-tu.cpp.json | FileCheck --match-full-lines -implicit-check-not='{{warning:|error:}}' -check-prefix=CHECK-FILE %s

// CHECK-CONSOLE-NOT: ===-------------------------------------------------------------------------===
// CHECK-CONSOLE-NOT: {{.*}}  --- Name ---
// CHECK-CONSOLE-NOT: {{.*}}  readability-function-size
// CHECK-CONSOLE-NOT: {{.*}}  Total
// CHECK-CONSOLE-NOT: ===-------------------------------------------------------------------------===

// CHECK-FILE: {
// CHECK-FILE-NEXT:"file": "{{.*}}clang-tidy-store-check-profile-one-tu.cpp",
// CHECK-FILE-NEXT:"timestamp": "{{[0-9]+}}-{{[0-9]+}}-{{[0-9]+}} {{[0-9]+}}:{{[0-9]+}}:{{[0-9]+}}.{{[0-9]+}}",
// CHECK-FILE-NEXT:"profile": {
// CHECK-FILE-NEXT:	"time.clang-tidy.readability-function-size.wall": {{.*}}{{[0-9]}}.{{[0-9]+}}e{{[-+]}}{{[0-9]}}{{[0-9]}},
// CHECK-FILE-NEXT:	"time.clang-tidy.readability-function-size.user": {{.*}}{{[0-9]}}.{{[0-9]+}}e{{[-+]}}{{[0-9]}}{{[0-9]}},
// CHECK-FILE-NEXT:	"time.clang-tidy.readability-function-size.sys": {{.*}}{{[0-9]}}.{{[0-9]+}}e{{[-+]}}{{[0-9]}}{{[0-9]}}{{,?}}
// If available on the platform, we also have a "time.clang-tidy.readability-function-size.instr" entry
// CHECK-FILE: }
// CHECK-FILE-NEXT: }

// CHECK-FILE-NOT: {
// CHECK-FILE-NOT: "file": {{.*}}clang-tidy-store-check-profile-one-tu.cpp{{.*}},
// CHECK-FILE-NOT: "timestamp": "{{[0-9]+}}-{{[0-9]+}}-{{[0-9]+}} {{[0-9]+}}:{{[0-9]+}}:{{[0-9]+}}.{{[0-9]+}}",
// CHECK-FILE-NOT: "profile": {
// CHECK-FILE-NOT:	"time.clang-tidy.readability-function-size.wall": {{.*}}{{[0-9]}}.{{[0-9]+}}e{{[-+]}}{{[0-9]}}{{[0-9]}},
// CHECK-FILE-NOT:	"time.clang-tidy.readability-function-size.user": {{.*}}{{[0-9]}}.{{[0-9]+}}e{{[-+]}}{{[0-9]}}{{[0-9]}},
// CHECK-FILE-NOT:	"time.clang-tidy.readability-function-size.sys": {{.*}}{{[0-9]}}.{{[0-9]+}}e{{[-+]}}{{[0-9]}}{{[0-9]}}{{,?}}
// CHECK-FILE-NOT: }
// CHECK-FILE-NOT: }

class A {
  A() {}
  ~A() {}
};
