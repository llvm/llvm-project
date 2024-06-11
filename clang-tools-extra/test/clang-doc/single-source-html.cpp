// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --format=html --executor=standalone -p %t %t/test.cpp -output=%t/docs > %t/output.txt
// RUN: cat %t/output.txt | FileCheck %s --check-prefix=CHECK

// CHECK: Using default asset: {{.*}}..\share\clang