// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc -output=%t %s 2>&1 | FileCheck %s --implicit-check-not="{{warning|error}}"

// COM: This case triggered an assertion before #141990:
// COM: clang-doc: llvm-project/clang/lib/AST/Decl.cpp:2985:
// COM:   Expr *clang::ParmVarDecl::getDefaultArg(): Assertion `!hasUninstantiatedDefaultArg()
// COM:   && "Default argument is not yet instantiated!"' failed.

template <class = int>
class c;
int e;

template <class>
class c {
public:
  void f(int n = e);
};
class B : c<> {};
