// RUN: %clang -std=c++98 %s -Wno-c++0x-compat -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX98 %s
// RUN: %clang -std=gnu++98 %s -Wno-c++0x-compat -fsyntax-only 2>&1 | FileCheck -check-prefix=GNUXX98 %s
// RUN: %clang -std=c++03 %s -Wno-c++0x-compat -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX98 %s
// RUN: %clang -std=c++0x %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX11 %s
// RUN: %clang -std=gnu++0x %s -fsyntax-only 2>&1 | FileCheck -check-prefix=GNUXX11 %s
// RUN: %clang -std=c++11 %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX11 %s
// RUN: %clang -std=gnu++11 %s -fsyntax-only 2>&1 | FileCheck -check-prefix=GNUXX11 %s
// RUN: %clang -std=c++1y %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX1Y %s
// RUN: %clang -std=gnu++1y %s -fsyntax-only 2>&1 | FileCheck -check-prefix=GNUXX1Y %s

void f(int n) {
  typeof(n)();
  decltype(n)();
}

// CXX98: undeclared identifier 'typeof'
// CXX98: undeclared identifier 'decltype'

// GNUXX98-NOT: undeclared identifier 'typeof'
// GNUXX98: undeclared identifier 'decltype'

// CXX11: undeclared identifier 'typeof'
// CXX11-NOT: undeclared identifier 'decltype'

// GNUXX11-NOT: undeclared identifier 'typeof'
// GNUXX11-NOT: undeclared identifier 'decltype'

// CXX1Y: undeclared identifier 'typeof'
// CXX1Y-NOT: undeclared identifier 'decltype'

// GNUXX1Y-NOT: undeclared identifier 'typeof'
// GNUXX1Y-NOT: undeclared identifier 'decltype'
