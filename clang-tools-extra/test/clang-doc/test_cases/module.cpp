// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

export module M;

int moduleFunction(int x); // ModuleLinkage

static int staticModuleFunction(int x); // ModuleInternalLinkage

export double exportedModuleFunction(double y, int z); // ExternalLinkage
