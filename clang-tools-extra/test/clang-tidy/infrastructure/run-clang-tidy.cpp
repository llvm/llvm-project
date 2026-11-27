// RUN: %run_clang_tidy --help
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %/t/test.cpp\",\"file\":\"%/t/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: echo "Checks: '-*,modernize-use-auto'" > %t/.clang-tidy
// RUN: echo "WarningsAsErrors: '*'" >> %t/.clang-tidy
// RUN: echo "CheckOptions:" >> %t/.clang-tidy
// RUN: echo "  modernize-use-auto.MinTypeNameLength: '0'" >> %t/.clang-tidy
// RUN: cp "%s" "%t/test.cpp"
// RUN: cd "%t"
// RUN: not %run_clang_tidy "test.cpp" 2>&1 | FileCheck %s --check-prefix=CHECK-JMAX
// CHECK-JMAX: Running clang-tidy in {{[1-9][0-9]*}} threads for

// RUN: not %run_clang_tidy -j 1 "test.cpp" 2>&1 | FileCheck %s --check-prefix=CHECK-J1
// CHECK-J1: Running clang-tidy in 1 threads for

int main()
{
  int* x = new int();
  delete x;
}
