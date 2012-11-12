// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %t/test.cpp\",\"file\":\"%t/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-check "%t/test.cpp" 2>&1 | FileCheck %s

// CHECK: C++ requires
invalid;

// REQUIRES: shell
