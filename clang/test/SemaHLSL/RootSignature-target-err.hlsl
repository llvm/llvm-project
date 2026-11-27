// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-rootsignature -hlsl-entry NotFoundRS -fsyntax-only %s -verify

// expected-error@* {{rootsignature specified as target environment but entry, NotFoundRS, was not defined}}

#define EntryRootSig "CBV(b0)"
