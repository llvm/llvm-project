// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s

IDENT.2
// CHECK: {{^}}IDENT.2{{$}}


// PR4395
#define X .*
X
// CHECK: {{^}}.*{{$}}

