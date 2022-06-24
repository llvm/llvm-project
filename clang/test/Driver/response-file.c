// Check that clang is able to process short response files
// Since this is a short response file, clang must not use a response file
// to pass its parameters to other tools. This is only necessary for a large
// number of parameters.
// RUN: echo "-DTEST" > %t.0.txt
// RUN: %clang -E @%t.0.txt %s -v 2>&1 | FileCheck %s -check-prefix=SHORT
// SHORT-NOT: Arguments passed via response file
// SHORT: extern int it_works;

// Check that clang is able to process long response files, routing a long
// sequence of arguments to other tools by using response files as well.
// We generate a 2MB response file to attempt to surpass any system limit.
// But there's no guarantee that we actually will (the system limit could be
// *huge*), so just check that invoking cc1 succeeds under these conditions.

// RUN: %python  -c 'print(*("-DTEST" for x in range(300000)))' >%t.1.txt
// RUN: %clang -E @%t.1.txt %s -v 2>&1 | FileCheck %s -check-prefix=LONG
// LONG: extern int it_works;

#ifdef TEST
extern int it_works;
#endif
