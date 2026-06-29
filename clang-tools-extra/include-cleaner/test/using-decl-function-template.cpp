// Regression test for a function-template named through a using-declaration.
//
// ns::ft is defined in func_template.h ("a.h"). We only #include the indirect
// func_template_reexport.h ("b.h"), bring ns::ft into scope with a local
// using-declaration, then call it. include-cleaner should report that the
// indirect header is stale (remove it) AND that the origin header is missing
// (insert it).
//
// BUG: for a *function template* (unlike a plain function, a class template, or
// a variable template) the use is recorded on the specialization / using-shadow
// decl, never on the primary FunctionTemplateDecl. So WalkAST's VisitUsingDecl
// downgrades the reference to the origin to RefType::Ambiguous, the origin
// header is never *demanded*, and include-cleaner removes the indirect header
// without inserting the origin -- a build-breaking suggestion.
//
// Once WalkAST.cpp treats a used function-template using-decl as an explicit
// reference, this should pass; drop the XFAIL below at that point.
//
// XFAIL: *

#include "func_template_reexport.h"

using ::ns::ft;

void use() { ft<int>(0); }

//      RUN: clang-include-cleaner -print=changes %s -- -I%S/Inputs/ | FileCheck %s
//      CHECK: - "func_template_reexport.h"
// CHECK-NEXT: + "func_template.h"
