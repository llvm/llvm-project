// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cd %t
// RUN: echo '[{"directory":".","command":"clang++ -c %t/test.cpp -o foo -ofoo","file":"%t/test.cpp"}]' | sed -e 's/\\/\//g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: echo '// CHECK: {{qwerty}}' > %t/cclog-check
// RUN: clang-check -p "%t" "%t/test.cpp" -analyze -analyzer-output-path=%t/qwerty -extra-arg=-v -extra-arg=-Xclang -extra-arg=-verify 2>&1 | FileCheck %t/cclog-check
// RUN: FileCheck %s --input-file=%t/qwerty

// CHECK: DOCTYPE plist
// CHECK: Division by zero
int f() {
  return 1 / 0; // expected-warning {{Division by zero}}
}
