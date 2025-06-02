// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fexperimental-bounds-safety-cxx -verify %s
//
// This test cases causes the `RD->isStruct() || RD->isUnion()` assert in
// `diagnoseDynamicCountVarInit` to fail because the type of
// `RefParamMustBePtrBad<int*> good0` is a `ClassTemplateSpecializationDecl`
// (i.e. not a struct or a union) (rdar://152452705). Even if that assert is
// removed the diagnostics won't be correct because support for `__single` and
// `__unsafe_indexable` in template definitions is not implemented in
// `-fbounds-safety` mode (rdar://152528102).
//
// XFAIL: *

// include other test to avoid duplicating test content for now.
#include "attributes_in_template_decls_attr_only_mode.cpp"
