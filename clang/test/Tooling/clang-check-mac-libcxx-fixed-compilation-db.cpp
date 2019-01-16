// Clang on MacOS can find libc++ living beside the installed compiler.
// This test makes sure our libTooling-based tools emulate this properly with
// fixed compilation database.
//
// RUN: rm -rf %t
// RUN: mkdir %t
//
// Install the mock libc++ (simulates the libc++ directory structure).
// RUN: cp -r %S/Inputs/mock-libcxx %t/
//
// RUN: cp $(which clang-check) %t/mock-libcxx/bin/
// RUN: cp "%s" "%t/test.cpp"
// RUN: %t/mock-libcxx/bin/clang-check -p "%t" "%t/test.cpp" -- -stdlib=libc++ -target x86_64-apple-darwin
// REQUIRES: system-darwin
#include <mock_vector>
vector v;
