// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s

#define FOO(x) x
#define BAR(x) x x
#define BAZ(x) [x] [ x] [x ]
[FOO()] [ FOO()] [FOO() ] [BAR()] [ BAR()] [BAR() ] BAZ()
// CHECK: [] [ ] [ ] [ ] [ ] [ ] [] [ ] [ ]
