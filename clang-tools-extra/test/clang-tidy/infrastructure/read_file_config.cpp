// REQUIRES: static-analyzer
// RUN: mkdir -p %t.dir/read-file-config/
// RUN: cp %s %t.dir/read-file-config/test.cpp
// RUN: echo 'Checks: "-*,modernize-use-nullptr"' > %t.dir/read-file-config/.clang-tidy
// RUN: echo '[{"command": "cc -c -o test.o test.cpp", "directory": "%/t.dir/read-file-config", "file": "%/t.dir/read-file-config/test.cpp"}]' > %t.dir/read-file-config/compile_commands.json
// RUN: clang-tidy %t.dir/read-file-config/test.cpp | not grep "warning: .*\[clang-analyzer-deadcode.DeadStores\]$"
// RUN: clang-tidy -checks="-*,clang-analyzer-*" %t.dir/read-file-config/test.cpp | grep "warning: .*\[clang-analyzer-deadcode.DeadStores\]$"

void f() {
  int x;
  x = 1;
  x = 2;
}
