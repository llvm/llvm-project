// RUN: not %clang -std=c++98 %s -Wno-c++0x-compat -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX98 %s
// RUN: not %clang -std=gnu++98 %s -Wno-c++0x-compat -fsyntax-only 2>&1 | FileCheck -check-prefix=GNUXX98 %s
// RUN: not %clang -std=c++03 %s -Wno-c++0x-compat -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX98 %s
// RUN: not %clang -std=c++0x %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX11 %s
// RUN: not %clang -std=gnu++0x %s -fsyntax-only 2>&1 | FileCheck -check-prefix=GNUXX11 %s
// RUN: not %clang -std=c++11 %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX11 %s
// RUN: not %clang -std=gnu++11 %s -fsyntax-only 2>&1 | FileCheck -check-prefix=GNUXX11 %s
// RUN: not %clang -std=c++1y %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX1Y %s
// RUN: not %clang -std=gnu++1y %s -fsyntax-only 2>&1 | FileCheck -check-prefix=GNUXX1Y %s
// RUN: not %clang -std=c++1z %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX1Z %s
// RUN: not %clang -std=gnu++1z %s -fsyntax-only 2>&1 | FileCheck -check-prefix=GNUXX1Z %s
// RUN: not %clang -std=c++2a %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CXX2A %s
// RUN: not %clang -std=gnu++2a %s -fsyntax-only 2>&1 | FileCheck -check-prefix=GNUXX2A %s


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

// CXX1Z: undeclared identifier 'typeof'
// CXX1Z-NOT: undeclared identifier 'decltype'

// GNUXX1Z-NOT: undeclared identifier 'typeof'
// GNUXX1Z-NOT: undeclared identifier 'decltype'

// CXX2A: undeclared identifier 'typeof'
// CXX2A-NOT: undeclared identifier 'decltype'

// GNUXX2A-NOT: undeclared identifier 'typeof'
// GNUXX2A-NOT: undeclared identifier 'decltype'

