// UNSUPPORTED: system-windows
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/level1/level2

// RUN: cd %t.dir
// RUN: echo "*" > .clang-format-ignore
// RUN: echo "level*/*.c*" >> .clang-format-ignore
// RUN: echo "*/*2/foo.*" >> .clang-format-ignore
// RUN: touch foo.cc
// RUN: clang-format -verbose .clang-format-ignore foo.cc 2>&1 \
// RUN:   | FileCheck %s -allow-empty

// RUN: cd level1
// RUN: touch bar.cc baz.c
// RUN: clang-format -verbose bar.cc baz.c 2>&1 | FileCheck %s -allow-empty

// RUN: cd level2
// RUN: touch foo.c foo.js
// RUN: clang-format -verbose foo.c foo.js 2>&1 | FileCheck %s -allow-empty

// CHECK-NOT: Formatting

// RUN: touch .clang-format-ignore
// RUN: clang-format -verbose foo.c foo.js 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK2 -match-full-lines
// CHECK2: Formatting [1/2] foo.c
// CHECK2-NEXT: Formatting [2/2] foo.js

// RUN: echo "*.js" > .clang-format-ignore
// RUN: clang-format -verbose foo.c foo.js 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK3 -match-full-lines
// CHECK3: Formatting [1/2] foo.c
// CHECK3-NOT: foo.js

// RUN: cd ../..
// RUN: clang-format -verbose *.cc level1/*.c* level1/level2/foo.* 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK4 -match-full-lines
// CHECK4: {{Formatting \[1/5] .*foo\.c}}
// CHECK4-NOT: foo.js

// RUN: rm .clang-format-ignore
// RUN: clang-format -verbose *.cc level1/*.c* level1/level2/foo.* 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK5 -match-full-lines
// CHECK5: {{Formatting \[1/5] .*foo\.cc}}
// CHECK5-NEXT: {{Formatting \[2/5] .*bar\.cc}}
// CHECK5-NEXT: {{Formatting \[3/5] .*baz\.c}}
// CHECK5-NEXT: {{Formatting \[4/5] .*foo\.c}}
// CHECK5-NOT: foo.js

// RUN: cd ..
// RUN: rm -r %t.dir
