// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %t/foo.cppm -emit-module-interface \
// RUN:   -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %t/user.cc -fmodule-file=foo=%t/foo.pcm \
// RUN:   -emit-llvm -o - -disable-llvm-passes | FileCheck %t/user.cc
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %t/user.cc -fmodule-file=foo=%t/foo.pcm \
// RUN:   -fexperimental-new-constant-interpreter -emit-llvm -o - -disable-llvm-passes | FileCheck %t/user.cc

//--- foo.cppm
// embed content

export module foo;

export constexpr const char data[] = {
#embed __FILE__ limit(16)
    , 0};

//--- user.cc
export module importer;
import foo;

constexpr auto d = data;
static_assert(d[0] == '/');

// CHECK: @{{.*}}data{{.*}} = {{.*}} constant [17 x i8] c"// embed content\00", align 16
