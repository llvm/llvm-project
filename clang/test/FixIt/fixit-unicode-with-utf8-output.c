// This test is an additional set of checks for the fixit-unicode.c test for
// systems capable of outputting Unicode characters to the standard output in
// the UTF-8 encoding.
// RUN: not %clang_cc1 -fsyntax-only -fno-diagnostics-show-line-numbers %S/fixit-unicode.c 2>&1 | FileCheck -strict-whitespace %s
// REQUIRES: utf8-capable-terminal

// CHECK: warning: format specifies type 'int' but the argument has type 'long'
// CHECK: {{^  printf\("∆: %d", 1L\);}}
// CHECK: {{^             ~~   \^~}}

// CHECK: error: use of undeclared identifier 'กsss'; did you mean 'กssss'?
// CHECK: {{^          \^}}
// CHECK: {{^          กssss}}

// CHECK: error: use of undeclared identifier 'ssกs'; did you mean 'ssกss'?
// CHECK: {{^          \^}}
// CHECK: {{^          ssกss}}

// CHECK: error: use of undeclared identifier 'ss一二三'; did you mean 's一二三'?
// CHECK: {{^                        \^~~~~~~~}}
// CHECK: {{^                        s一二三}}

// CHECK: error: use of undeclared identifier 'sssssssss'; did you mean 'sssssssssก'?
// CHECK: {{^          \^}}
// CHECK: {{^          sssssssssก}}
