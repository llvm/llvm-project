// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -DSECRET=XYZZY -c %/t/test.cpp\",\"file\":\"%/t/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: not clang-check "%t/test.cpp" 2>&1 | FileCheck %s

// CHECK: XYZZY
SECRET;

// Copy to a different file, and rely on the command being inferred.
// RUN: cp "%s" "%t/other.cpp"
// RUN: not clang-check "%t/other.cpp" 2>&1 | FileCheck %s
