// RUN: not clang-rename rename-all -new-name=Foo -old-name=Bar %s -- 2>&1 | FileCheck %s
// CHECK: clang-rename: could not find symbol Bar.
