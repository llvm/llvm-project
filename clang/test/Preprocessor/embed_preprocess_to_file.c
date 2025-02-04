// RUN: %clang_cc1 -std=c23 %s -E --embed-dir=%S/Inputs | FileCheck %s --check-prefix EXPANDED
// RUN: %clang_cc1 -std=c23 %s -E -dE --embed-dir=%S/Inputs | FileCheck %s --check-prefix DIRECTIVE

// Ensure that we correctly preprocess to a file, both with expanding embed
// directives fully and with printing the directive instead.
const char data[] = {
#embed <jk.txt> if_empty('a', 'b') clang::offset(0) limit(1) suffix(, 'a', 0) prefix('h',)
};

// EXPANDED: const char data[] = {'h',106 , 'a', 0};
// DIRECTIVE: const char data[] = {
// DIRECTIVE-NEXT: #embed <jk.txt> if_empty('a', 'b') limit(1) clang::offset(0) prefix('h',) suffix(, 'a', 0) /* clang -E -dE */
// DIRECTIVE-NEXT: };

const char more[] = {
#embed <media/empty> if_empty('a', 'b')
};

// EXPANDED: const char more[] = {'a', 'b'}
// DIRECTIVE: const char more[] = {
// DIRECTIVE-NEXT: #embed <media/empty> if_empty('a', 'b') /* clang -E -dE */
// DIRECTIVE-NEXT: };

const char even_more[] = {
  1, 2, 3,
#embed <jk.txt> prefix(4, 5,) suffix(, 6, 7)
  , 8, 9, 10
};

// EXPANDED: const char even_more[] = {
// EXPANDED-NEXT:   1, 2, 3,4, 5,106, 107 , 6, 7 , 8, 9, 10
// EXPANDED-EMPTY:
// EXPANDED-EMPTY:
// EXPANDED-NEXT: };
// DIRECTIVE: const char even_more[] = {
// DIRECTIVE-NEXT:  1, 2, 3,
// DIRECTIVE-NEXT: #embed <jk.txt> prefix(4, 5,) suffix(, 6, 7) /* clang -E -dE */
// DIRECTIVE-NEXT:  , 8, 9, 10
// DIRECTIVE-NEXT: };
