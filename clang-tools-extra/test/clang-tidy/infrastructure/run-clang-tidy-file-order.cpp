// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %/t/a-small.cpp\",\"file\":\"%/t/a-small.cpp\"},{\"directory\":\".\",\"command\":\"clang++ -c %/t/b-large.cpp\",\"file\":\"%/t/b-large.cpp\"},{\"directory\":\".\",\"command\":\"clang++ -c %/t/c-small.cpp\",\"file\":\"%/t/c-small.cpp\"},{\"directory\":\".\",\"command\":\"clang++ -DTWO -c %/t/c-small.cpp\",\"file\":\"%/t/c-small.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: echo "Checks: '-*,misc-unused-using-decls'" > %t/.clang-tidy
// RUN: echo "int a;" > %t/a-small.cpp
// RUN: cp "%s" "%t/b-large.cpp"
// RUN: echo "int c;" > %t/c-small.cpp
// RUN: cd "%t"
// RUN: %run_clang_tidy -j 1 2>&1 | FileCheck %s

// Files start largest first, counting a file once for each of its compile
// commands, and files of the same size in path order.
// CHECK: [1/3]{{.*}}b-large.cpp
// CHECK: [2/3]{{.*}}c-small.cpp
// CHECK: [3/3]{{.*}}a-small.cpp
