// RUN: %clang_cc1 -x c -std=c18 -fsyntax-only -pedantic -verify=expected,trigraphs,no-newline %s
// RUN: %clang_cc1 -x c -std=c23 -fsyntax-only -pedantic -verify=expected,no-trigraphs,c23 %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -std=c++98 -pedantic -verify=expected,trigraphs,no-newline %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -std=c++11 -Wc++98-compat-pedantic -verify=expected,trigraphs,no-newline-compat %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -std=c++11 -verify=expected,trigraphs %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -std=c++17 -verify=expected,no-trigraphs,cxx17 %s

// trigraphs-warning@+10 {{trigraph converted to '\' character}}
// no-newline-warning@+9 {{no newline at end of file}}
// no-newline-note@+8 {{last newline deleted by splice here}}
// no-newline-compat-warning@+7 {{C++98 requires newline at end of file}}
// no-newline-compat-note@+6 {{last newline deleted by splice here}}
// no-trigraphs-warning@+5 {{trigraph ignored}}
// c23-error@+4 {{expected identifier or '('}}
// cxx17-error@+3 {{expected unqualified-id}}
// trigraphs-warning@+2 {{backslash and newline separated by space}}
// The next line intentionally has a trailing tab character.
int x; ??/	
