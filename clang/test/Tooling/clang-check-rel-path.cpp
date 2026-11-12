// This block test a compilation database with files relative to the directory
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo '[{"directory":"%t","command":"clang++ -c test.cpp","file":"test.cpp"}]' | sed -e 's/\\/\//g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: not clang-check -p "%t" "%t/test.cpp" 2>&1|FileCheck %s
// FIXME: Make the above easier.

// CHECK: a type specifier is required
invalid;
