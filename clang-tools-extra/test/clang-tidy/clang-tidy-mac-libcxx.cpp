// Clang on MacOS can find libc++ living beside the installed compiler.
// This test makes sure clang-tidy emulates this properly.
//
// RUN: rm -rf %t
// RUN: mkdir %t
//
// Install the mock libc++ (simulates the libc++ directory structure).
// RUN: cp -r %S/Inputs/mock-libcxx %t/
//
// Pretend clang is installed beside the mock library that we provided.
// RUN: echo '[{"directory":"%t","command":"%t/mock-libcxx/bin/clang++ -stdlib=libc++ -std=c++11 -target x86_64-apple-darwin -c test.cpp","file":"test.cpp"}]' | sed -e 's/\\/\//g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-tidy -header-filter='.*' -system-headers -checks='-*,modernize-use-using'  "%t/test.cpp" | FileCheck %s
// CHECK: mock_vector:{{[0-9]+}}:{{[0-9]+}}: warning: use 'using' instead of 'typedef'

#include <mock_vector>
typedef vector* vec_ptr;
