// REQUIRES: static-analyzer
// RUN: mkdir -p %T/read-file-config/
// RUN: cp %s %T/read-file-config/test.cpp
// RUN: echo 'Checks: "-*,modernize-use-nullptr"' > %T/read-file-config/.clang-tidy
// RUN: echo '[{"command": "cc -c -o test.o test.cpp", "directory": "%/T/read-file-config", "file": "%/T/read-file-config/test.cpp"}]' > %T/read-file-config/compile_commands.json
// RUN: clang-tidy %T/read-file-config/test.cpp | not grep "warning: .*\[clang-analyzer-deadcode.DeadStores\]$"
// RUN: clang-tidy -checks="-*,clang-analyzer-*" %T/read-file-config/test.cpp | grep "warning: .*\[clang-analyzer-deadcode.DeadStores\]$"

void f() {
  int x;
  x = 1;
  x = 2;
}
