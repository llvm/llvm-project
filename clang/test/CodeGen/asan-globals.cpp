// RUN: echo "int extra_global;" > %t.extra-source.cpp
// RUN: echo "global:*blacklisted_global*" > %t.blacklist
// RUN: %clang_cc1 -include %t.extra-source.cpp -fsanitize=address -fsanitize-blacklist=%t.blacklist -emit-llvm -o - %s | FileCheck %s
// The blacklist file uses regexps, so Windows path backslashes.
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t.blacklist-src
// RUN: %clang_cc1 -include %t.extra-source.cpp -fsanitize=address -fsanitize-blacklist=%t.blacklist-src -emit-llvm -o - %s | FileCheck %s --check-prefix=BLACKLIST-SRC

int global;
int dyn_init_global = global;
int __attribute__((no_sanitize("address"))) attributed_global;
int blacklisted_global;

void func() {
  static int static_var = 0;
  const char *literal = "Hello, world!";
}

// CHECK: !llvm.asan.globals = !{![[EXTRA_GLOBAL:[0-9]+]], ![[GLOBAL:[0-9]+]], ![[DYN_INIT_GLOBAL:[0-9]+]], ![[ATTR_GLOBAL:[0-9]+]], ![[BLACKLISTED_GLOBAL:[0-9]+]], ![[STATIC_VAR:[0-9]+]], ![[LITERAL:[0-9]+]]}
// CHECK: ![[EXTRA_GLOBAL]] = !{{{.*}} ![[EXTRA_GLOBAL_LOC:[0-9]+]], !"extra_global", i1 false, i1 false}
// CHECK: ![[EXTRA_GLOBAL_LOC]] = !{!"{{.*}}extra-source.cpp", i32 1, i32 5}
// CHECK: ![[GLOBAL]] = !{{{.*}} ![[GLOBAL_LOC:[0-9]+]], !"global", i1 false, i1 false}
// CHECK: ![[GLOBAL_LOC]] = !{!"{{.*}}asan-globals.cpp", i32 8, i32 5}
// CHECK: ![[DYN_INIT_GLOBAL]] = !{{{.*}} ![[DYN_INIT_LOC:[0-9]+]], !"dyn_init_global", i1 true, i1 false}
// CHECK: ![[DYN_INIT_LOC]] = !{!"{{.*}}asan-globals.cpp", i32 9, i32 5}
// CHECK: ![[ATTR_GLOBAL]] = !{{{.*}}, null, null, i1 false, i1 true}
// CHECK: ![[BLACKLISTED_GLOBAL]] = !{{{.*}}, null, null, i1 false, i1 true}
// CHECK: ![[STATIC_VAR]] = !{{{.*}} ![[STATIC_LOC:[0-9]+]], !"static_var", i1 false, i1 false}
// CHECK: ![[STATIC_LOC]] = !{!"{{.*}}asan-globals.cpp", i32 14, i32 14}
// CHECK: ![[LITERAL]] = !{{{.*}} ![[LITERAL_LOC:[0-9]+]], !"<string literal>", i1 false, i1 false}
// CHECK: ![[LITERAL_LOC]] = !{!"{{.*}}asan-globals.cpp", i32 15, i32 25}

// BLACKLIST-SRC: !llvm.asan.globals = !{![[EXTRA_GLOBAL:[0-9]+]], ![[GLOBAL:[0-9]+]], ![[DYN_INIT_GLOBAL:[0-9]+]], ![[ATTR_GLOBAL:[0-9]+]], ![[BLACKLISTED_GLOBAL:[0-9]+]], ![[STATIC_VAR:[0-9]+]], ![[LITERAL:[0-9]+]]}
// BLACKLIST-SRC: ![[EXTRA_GLOBAL]] = !{{{.*}} ![[EXTRA_GLOBAL_LOC:[0-9]+]], !"extra_global", i1 false, i1 false}
// BLACKLIST-SRC: ![[EXTRA_GLOBAL_LOC]] = !{!"{{.*}}extra-source.cpp", i32 1, i32 5}
// BLACKLIST-SRC: ![[GLOBAL]] = !{{{.*}} null, null, i1 false, i1 true}
// BLACKLIST-SRC: ![[DYN_INIT_GLOBAL]] = !{{{.*}} null, null, i1 true, i1 true}
// BLACKLIST-SRC: ![[ATTR_GLOBAL]] = !{{{.*}}, null, null, i1 false, i1 true}
// BLACKLIST-SRC: ![[BLACKLISTED_GLOBAL]] = !{{{.*}}, null, null, i1 false, i1 true}
// BLACKLIST-SRC: ![[STATIC_VAR]] = !{{{.*}} null, null, i1 false, i1 true}
// BLACKLIST-SRC: ![[LITERAL]] = !{{{.*}} null, null, i1 false, i1 true}
