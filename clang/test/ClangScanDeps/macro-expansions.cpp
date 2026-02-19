// This checks that there's no issue with the preprocessor handling user or built-in macro
// expansion during dependency scanning.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json | FileCheck %s %if system-darwin && target={{.*}}-{{darwin|macos}}{{.*}} %{ --check-prefixes=CHECK,CHECK-DARWIN %}

// CHECK: test.o:
// CHECK-DARWIN-NEXT: SDKSettings.json
// CHECK-NEXT: test.cpp
// CHECK-NEXT: header1.h
// CHECK-NEXT: header2.h

//--- cdb.json.template
[{
  "directory" : "DIR",
  "command" : "clang -target x86_64-apple-macosx10.7 -c DIR/test.cpp -o DIR/test.o",
  "file" : "DIR/test.o"
}]

//--- test.cpp
#define FN_MACRO(x) 1
#if FN_MACRO(a)
#include "header1.h"
#endif

#if __has_cpp_attribute(clang::fallthrough)
#include "header2.h"
#endif

//--- header1.h
#ifndef _HEADER1_H_
#define _HEADER1_H_
#endif

//--- header2.h
#ifndef _HEADER2_H_
#define _HEADER2_H_
#endif
