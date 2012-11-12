// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// Test for absence of crash reported in PR 2923:
//
//  http://llvm.org/bugs/show_bug.cgi?id=2923
//
// Previously we had a crash when deallocating the FunctionDecl for 'bar'
// because FunctionDecl::getNumParams() just used the type of foo to determine
// the number of parameters it has.  In the case of 'bar' there are no
// ParmVarDecls.
int foo(int x, int y) { return x + y; }
extern typeof(foo) bar;
