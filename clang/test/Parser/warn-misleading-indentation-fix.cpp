int main(void) { for( int i = 0; i < 9; i++ ); return 0; }
// the crash only occurs on the first line don't move it.
// RUN: %clang_cc1 -x c -fsyntax-only -Wmisleading-indentation %s
