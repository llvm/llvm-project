// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s

#if 0
blah
#\
else
bark
#endif
// CHECK: {{^}}bark{{$}}

