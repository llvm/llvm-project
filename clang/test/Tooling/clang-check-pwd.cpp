// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %t/test.cpp\",\"file\":\"%t/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: ln -sf %t %t.foobar
// RUN: cd %t
// RUN: env PWD="%t.foobar" not clang-check -p "%t" "test.cpp" 2>&1|FileCheck %s
// FIXME: Make the above easier.

// CHECK: C++ requires
// CHECK: .foobar/test.cpp
invalid;

// REQUIRES: shell
