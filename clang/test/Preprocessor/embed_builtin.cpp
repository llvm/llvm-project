// RUN: %clang_cc1 %s -fsyntax-only -embed-dir=%S/Inputs -verify
// expected-no-diagnostics

#if !__has_builtin(__builtin_pp_embed)
#error "Don't have __builtin_pp_embed?"
#endif
