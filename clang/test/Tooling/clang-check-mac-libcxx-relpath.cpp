// Clang on MacOS can find libc++ living beside the installed compiler.
// This test makes sure our libTooling-based tools emulate this properly.
//
// RUN: rm -rf %t
// RUN: mkdir %t
//
// Install the mock libc++ (simulates the libc++ directory structure).
// RUN: cp -r %S/Inputs/mock-libcxx %t/
//
// Pretend clang is installed beside the mock library that we provided.
// RUN: echo '[{"directory":"%t","command":"mock-libcxx/bin/clang++ -stdlib=libc++ -target x86_64-apple-darwin -c test.cpp","file":"test.cpp"}]' | sed -e 's/\\/\//g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// clang-check will produce an error code if the mock library is not found.
// RUN: clang-check -p "%t" "%t/test.cpp"

#include <mock_vector>
vector v;
