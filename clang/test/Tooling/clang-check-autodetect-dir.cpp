// RUN: rm -rf %t
// RUN: mkdir -p %t/abc/def
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %t/test.cpp\",\"file\":\"%t/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-check -p "%t/abc/def" "%t/test.cpp" 2>&1|FileCheck %s
// FIXME: Make the above easier.

// CHECK: C++ requires
invalid;

// REQUIRES: shell
