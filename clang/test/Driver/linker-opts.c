// RUN: env LIBRARY_PATH=%T/test1 %clang -x c %s -### 2>&1 | FileCheck %s
// CHECK: "-L{{.*}}/test1"

// GCC driver is used as linker on cygming. It should be aware of LIBRARY_PATH.
// XFAIL: cygwin,mingw32,win32
