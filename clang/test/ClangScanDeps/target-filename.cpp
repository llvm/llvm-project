// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/target-filename_input.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/target-filename-cdb.json > %t.cdb
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 | FileCheck %s --check-prefixes=CHECK,%if system-darwin  && target={{.*}}-{{darwin|macos}}{{.*}} %{CHECK-DARWIN %} %else %{CHECK-NON-DARWIN %}

// CHECK: target-filename_input.o:
// CHECK-DARWIN-NEXT: SDKSettings.json
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NON-DARWIN-NEXT: a.o:
// CHECK-DARWIN-NEXT: a.o:{{.*[[:space:]].*}}SDKSettings.json
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NON-DARWIN-NEXT: b.o:
// CHECK-DARWIN-NEXT: b.o:{{.*[[:space:]].*}}SDKSettings.json
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: last.o:
// CHECK-DARWIN-NEXT: SDKSettings.json
// CHECK-NEXT: target-filename_input.cpp

// CHECK: target-filename_input.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: clangcl-a.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: clangcl-b.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: clangcl-c.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: clangcl-d.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: clangcl-e.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: clangcl-lastf.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: clangcl-lastg.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: clangcl-lasth.o:
// CHECK-NEXT: target-filename_input.cpp
