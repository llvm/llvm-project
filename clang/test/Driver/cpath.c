// RUN: mkdir -p %t/test1 %t/test2 %t/test3

// RUN: env "CPATH=%t/test1%{pathsep}%t/test2" %clang -x c -E -v %s 2>&1 | FileCheck %s -check-prefix=CPATH
// CPATH: -I{{.*}}/test1
// CPATH: -I{{.*}}/test2
// CPATH: search starts here
// CPATH: test1
// CPATH: test2

// RUN: env "OBJC_INCLUDE_PATH=%t/test1%{pathsep}%t/test2"  OBJCPLUS_INCLUDE_PATH=%t/test1  "CPLUS_INCLUDE_PATH=%t/test1%{pathsep}%t/test2" C_INCLUDE_PATH=%t/test3 %clang -x c -E -v %s 2>&1 | FileCheck %s -check-prefix=C_INCLUDE_PATH
// C_INCLUDE_PATH: -c-isystem {{"?.*}}/test3{{"?}} -cxx-isystem {{"?.*}}/test1{{"?}} -cxx-isystem {{"?.*}}/test2{{"?}} -objc-isystem {{"?.*}}/test1{{"?}} -objc-isystem {{"?.*}}/test2{{"?}} -objcxx-isystem {{"?.*}}/test1{{"?}}
// C_INCLUDE_PATH: search starts here
// C_INCLUDE_PATH-NOT: test1
// C_INCLUDE_PATH: test3
// C_INCLUDE_PATH-NOT: test1

// RUN: env OBJC_INCLUDE_PATH=%t/test1 OBJCPLUS_INCLUDE_PATH=%t/test3 CPLUS_INCLUDE_PATH=%t/test3 C_INCLUDE_PATH=%t/test1 %clang -x objective-c++ -E -v %s 2>&1 | FileCheck %s -check-prefix=OBJCPLUS_INCLUDE_PATH
// OBJCPLUS_INCLUDE_PATH: -c-isystem {{"?.*}}/test1{{"?}} -cxx-isystem {{"?.*}}/test3{{"?}} -objc-isystem {{"?.*}}/test1{{"?}} -objcxx-isystem {{"?.*}}/test3{{"?}}
// OBJCPLUS_INCLUDE_PATH: search starts here
// OBJCPLUS_INCLUDE_PATH-NOT: test1
// OBJCPLUS_INCLUDE_PATH: test3
// OBJCPLUS_INCLUDE_PATH-NOT: test1
