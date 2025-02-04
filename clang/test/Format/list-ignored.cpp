// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/level1/level2

// RUN: cd %t.dir
// RUN: echo "*" > .clang-format-ignore
// RUN: echo "level*/*.c*" >> .clang-format-ignore
// RUN: echo "*/*2/foo.*" >> .clang-format-ignore

// RUN: touch foo.cc
// RUN: clang-format -list-ignored .clang-format-ignore foo.cc \
// RUN:   | FileCheck %s
// CHECK: .clang-format-ignore
// CHECK-NEXT: foo.cc

// RUN: cd level1
// RUN: touch bar.cc baz.c
// RUN: clang-format -list-ignored bar.cc baz.c \
// RUN:   | FileCheck %s -check-prefix=CHECK2
// CHECK2: bar.cc
// CHECK2-NEXT: baz.c

// RUN: cd level2
// RUN: touch foo.c foo.js
// RUN: clang-format -list-ignored foo.c foo.js \
// RUN:   | FileCheck %s -check-prefix=CHECK3
// CHECK3: foo.c
// CHECK3-NEXT: foo.js

// RUN: touch .clang-format-ignore
// RUN: clang-format -list-ignored foo.c foo.js \
// RUN:   | FileCheck %s -allow-empty -check-prefix=CHECK4
// CHECK4-NOT: foo.c
// CHECK4-NOT: foo.js

// RUN: echo "*.js" > .clang-format-ignore
// RUN: clang-format -list-ignored foo.c foo.js \
// RUN:   | FileCheck %s -check-prefix=CHECK5
// CHECK5-NOT: foo.c
// CHECK5: foo.js

// RUN: cd ../..
// RUN: clang-format -list-ignored *.cc level1/*.c* level1/level2/foo.* \
// RUN:   | FileCheck %s -check-prefix=CHECK6
// CHECK6: foo.cc
// CHECK6-NEXT: bar.cc
// CHECK6-NEXT: baz.c
// CHECK6-NOT: foo.c
// CHECK6-NEXT: foo.js

// RUN: rm .clang-format-ignore
// RUN: clang-format -list-ignored *.cc level1/*.c* level1/level2/foo.* \
// RUN:   | FileCheck %s -check-prefix=CHECK7
// CHECK7-NOT: foo.cc
// CHECK7-NOT: bar.cc
// CHECK7-NOT: baz.c
// CHECK7-NOT: foo.c
// CHECK7: foo.js

// RUN: cd ..
// RUN: rm -r %t.dir
