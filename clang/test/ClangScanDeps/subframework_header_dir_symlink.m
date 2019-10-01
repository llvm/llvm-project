// REQUIRES: shell
// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/subframework_header_dir_symlink_input.m
// RUN: cp %s %t.dir/subframework_header_dir_symlink_input2.m
// RUN: mkdir %t.dir/Inputs
// RUN: cp -R %S/Inputs/frameworks %t.dir/Inputs/frameworks
// RUN: ln -s %t.dir/Inputs/frameworks %t.dir/Inputs/frameworks_symlink
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/subframework_header_dir_symlink_cdb.json > %t.cdb
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -reuse-filemanager=0 | \
// RUN:   FileCheck %s
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -reuse-filemanager=1 | \
// RUN:   FileCheck %s

#ifndef EMPTY
#include "Framework/Framework.h"
#endif

// CHECK: subframework_header_dir_symlink_input.o
// CHECK-NEXT: subframework_header_dir_symlink_input.m
// CHECK: subframework_header_dir_symlink_input2.o
// CHECK-NEXT: subframework_header_dir_symlink_input2.m
// CHECK-NEXT: Inputs{{/|\\}}frameworks_symlink{{/|\\}}Framework.framework{{/|\\}}Headers{{/|\\}}Framework.h
