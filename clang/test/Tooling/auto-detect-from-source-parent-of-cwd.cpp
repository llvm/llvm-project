// RUN: rm -rf %t
// RUN: mkdir -p %t/abc/def/ijk/qwe
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %t/abc/def/ijk/qwe/test.cpp\",\"file\":\"%t/abc/def/ijk/qwe/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: cp "%s" "%t/abc/def/ijk/qwe/test.cpp"
// RUN: PWD="%t/abc/def" clang-check "ijk/qwe/test.cpp" 2>&1 | FileCheck %s

// CHECK: C++ requires
invalid;

// REQUIRES: shell
