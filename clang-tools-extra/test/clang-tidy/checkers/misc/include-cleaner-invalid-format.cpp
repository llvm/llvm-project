// RUN: mkdir -p %t.dir
// RUN: echo 'IndentWidth: 2' > %t.dir/.clang-format
// RUN: echo 'IndentWidth: 4' >> %t.dir/.clang-format
// RUN: %check_clang_tidy %s misc-include-cleaner %t.dir/repro

// CHECK-MESSAGES: error: duplicated mapping key 'IndentWidth'
