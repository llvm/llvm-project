// under test:
// - .clang-tidy read from file & treat warnings as errors
// - return code of run-clang-tidy on those errors

// First make sure clang-tidy is executable and can print help without crashing:
// RUN: %run_clang_tidy --help

// use %t as directory instead of file:
// RUN: rm -rf %t
// RUN: mkdir %t

// add this file to %t, add compile_commands for it and .clang-tidy config:
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %/t/test.cpp\",\"file\":\"%/t/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: echo "Checks: '-*,modernize-use-auto'" > %t/.clang-tidy
// RUN: echo "WarningsAsErrors: '*'" >> %t/.clang-tidy
// RUN: echo "CheckOptions:" >> %t/.clang-tidy
// RUN: echo "  - key:             modernize-use-auto.MinTypeNameLength" >> %t/.clang-tidy
// RUN: echo "    value:           '0'" >> %t/.clang-tidy
// RUN: cp "%s" "%t/test.cpp"

// execute and check:
// RUN: cd "%t"
// RUN: not %run_clang_tidy "test.cpp" > %t/msg.txt 2>&1
// RUN: FileCheck -input-file=%t/msg.txt -check-prefix=CHECK-MESSAGES %s \
// RUN:           -implicit-check-not='{{warning|error|note}}:'

int main()
{
  int* x = new int();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: error: {{.+}} [modernize-use-auto,-warnings-as-errors]

  delete x;
}
