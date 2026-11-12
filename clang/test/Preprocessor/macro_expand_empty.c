// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s

// Check that this doesn't crash

#define IDENTITY1(x) x
#define IDENTITY2(x) IDENTITY1(x) IDENTITY1(x) IDENTITY1(x) IDENTITY1(x)
#define IDENTITY3(x) IDENTITY2(x) IDENTITY2(x) IDENTITY2(x) IDENTITY2(x)
#define IDENTITY4(x) IDENTITY3(x) IDENTITY3(x) IDENTITY3(x) IDENTITY3(x)
#define IDENTITY5(x) IDENTITY4(x) IDENTITY4(x) IDENTITY4(x) IDENTITY4(x)
#define IDENTITY6(x) IDENTITY5(x) IDENTITY5(x) IDENTITY5(x) IDENTITY5(x)
#define IDENTITY7(x) IDENTITY6(x) IDENTITY6(x) IDENTITY6(x) IDENTITY6(x)
#define IDENTITY8(x) IDENTITY7(x) IDENTITY7(x) IDENTITY7(x) IDENTITY7(x)
#define IDENTITY9(x) IDENTITY8(x) IDENTITY8(x) IDENTITY8(x) IDENTITY8(x)
#define IDENTITY0(x) IDENTITY9(x) IDENTITY9(x) IDENTITY9(x) IDENTITY9(x)
IDENTITY0()

#define FOO() BAR() second
#define BAR()
first // CHECK: {{^}}first{{$}}
FOO() // CHECK: {{^}} second{{$}}
third // CHECK: {{^}}third{{$}}
