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

// RUN: rm -rf %t-matching
// RUN: mkdir -p %t-matching/include
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %/t-matching/test.cpp -I%/t-matching/include\",\"file\":\"%/t-matching/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t-matching/compile_commands.json
// RUN: echo "Checks: '-*,misc-confusable-identifiers'" > %t-matching/.clang-tidy
// RUN: echo "WarningsAsErrors: '*'" >> %t-matching/.clang-tidy
// RUN: echo "int l0 = 0;" > %t-matching/include/confusable_header.h
// RUN: echo '#include "confusable_header.h"' > %t-matching/test.cpp
// RUN: echo 'int lO = 1;' >> %t-matching/test.cpp
// RUN: cd "%t-matching"
// RUN: not %run_clang_tidy -j 1 -header-filter=does-not-match "test.cpp" 2>&1 | FileCheck %s --check-prefix=CHECK-MATCHING-OFF
// RUN: %run_clang_tidy -j 1 -header-filter=does-not-match --experimental-header-filter-matching=true "test.cpp" 2>&1 | FileCheck %s --check-prefix=CHECK-MATCHING-ON
// CHECK-MATCHING-OFF: 'lO' is confusable with 'l0'
// CHECK-MATCHING-ON: Running clang-tidy in 1 threads for 1 files out of 1 in compilation database
// CHECK-MATCHING-ON-NOT: 'lO' is confusable with 'l0'

int main()
{
  int* x = new int();
  delete x;
}
