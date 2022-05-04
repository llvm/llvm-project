// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo '[{"directory":".","command":"clang++ -c %t/test.cpp -o foo -ofoo","file":"%t/test.cpp"}]' | sed -e 's/\\/\//g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: not clang-check -p "%t" "%t/test.cpp" -extra-arg=-v 2>&1|FileCheck %s
// FIXME: Make the above easier.

// CHECK: Invocation
// CHECK-NOT: {{ -v}}
// CHECK: a type specifier is required
invalid;
