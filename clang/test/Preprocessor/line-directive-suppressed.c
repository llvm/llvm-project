// RUN: %clang_cc1 -std=c99 -fsyntax-only -pedantic %s 2>&1 | grep 'warning: this style of line directive is a GNU extension'

// RUN: %clang_cc1 -std=c99 -fsyntax-only -pedantic -x c-cpp-output %s 2>&1 | not grep warning
// RUN: cp %s %t.i
// RUN: %clang_cc1 -std=c99 -fsyntax-only -pedantic %t.i 2>&1 | not grep warning

# 0 "zero"
# 1 "one" 1
# 2 "two" 1 3 4
