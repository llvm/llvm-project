// REQUIRES: osx-ld64-live_support

// Compiling with PGO/code coverage on Darwin should raise no warnings or errors
// when using an exports list.

// RUN: echo "_main" > %t.exports
// RUN: %clang_pgogen -Werror -Wl,-exported_symbols_list,%t.exports -o %t %s 2>&1 | tee %t.log
// RUN: %clang_profgen -Werror -fcoverage-mapping -Wl,-exported_symbols_list,%t.exports -o %t %s 2>&1 | tee -a %t.log
// RUN: cat %t.log | count 0

int main() {}
