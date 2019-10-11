// RUN: %run_clang_tidy --help
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %/t/test.cpp\",\"file\":\"%/t/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: echo "Checks: '-*,modernize-use-auto'" > %t/.clang-tidy
// RUN: echo "WarningsAsErrors: '*'" >> %t/.clang-tidy
// RUN: echo "CheckOptions:" >> %t/.clang-tidy
// RUN: echo "  - key:             modernize-use-auto.MinTypeNameLength" >> %t/.clang-tidy
// RUN: echo "    value:           '0'" >> %t/.clang-tidy
// RUN: cp "%s" "%t/test.cpp"
// RUN: cd "%t"
// RUN: not %run_clang_tidy "%t/test.cpp"

int main()
{
  int* x = new int();
  delete x;
}
