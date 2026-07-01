// Regression test for a function-template named through a using-declaration.
//
// ns::ft is defined in func_template.h ("a.h"). We only #include the indirect
// func_template_reexport.h ("b.h"), bring ns::ft into scope with a local
// using-declaration, then call it. include-cleaner should report that the
// indirect header is stale (remove it) AND that the origin header is missing
// (insert it).
//
// Regression: for a *function template* (unlike a plain function, a class
// template, or a variable template) the call is recorded on the implicitly
// instantiated specialization, not on the primary FunctionTemplateDecl. If
// WalkAST's VisitUsingDecl only inspected the primary decl it would downgrade
// the reference to the origin to RefType::Ambiguous, never *demand* the origin
// header, and remove the indirect header without inserting the origin -- a
// build-breaking suggestion. VisitUsingDecl consults the specializations to
// avoid this.

#include "func_template_reexport.h"

using ::ns::ft;

void use() { ft<int>(0); }

//      RUN: clang-include-cleaner -print=changes %s -- -I%S/Inputs/ | FileCheck %s
//      CHECK: - "func_template_reexport.h"
// CHECK-NEXT: + "func_template.h"
