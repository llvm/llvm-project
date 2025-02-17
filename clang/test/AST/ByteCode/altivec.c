// RUN: %clang_cc1 -triple=powerpc64-unknown-linux-gnu -target-feature +altivec -target-feature +vsx -fsyntax-only -verify=expected,both %s
// RUN: %clang_cc1 -triple=powerpc64le-unknown-linux-gnu -target-feature +altivec -target-feature -vsx -fsyntax-only -verify=expected,both %s
// RUN: %clang_cc1 -triple=powerpc-ibm-aix -target-feature +altivec -fsyntax-only -verify=expected,both %s
// RUN: %clang_cc1 -triple=powerpc64-ibm-aix -target-feature +altivec -target-feature -vsx -fsyntax-only -verify=expected,both %s

// RUN: %clang_cc1 -triple=powerpc64-unknown-linux-gnu -target-feature +altivec -target-feature +vsx -fsyntax-only -verify=expected,both -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -triple=powerpc64le-unknown-linux-gnu -target-feature +altivec -target-feature -vsx -fsyntax-only -verify=expected,both -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -triple=powerpc-ibm-aix -target-feature +altivec -fsyntax-only -verify=expected,both -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -triple=powerpc64-ibm-aix -target-feature +altivec -target-feature -vsx -fsyntax-only -verify=expected,both -fexperimental-new-constant-interpreter %s

// both-no-diagnostics.

/// From test/Parser/altivec.c
vector char v1 = (vector char)((vector int)(1, 2, 3, 4));
vector char v2 = (vector char)((vector float)(1.0f, 2.0f, 3.0f, 4.0f));
vector char v3 = (vector char)((vector int)('a', 'b', 'c', 'd'));
vector int v4 = (vector int)(1, 2, 3, 4);
vector float v5 = (vector float)(1.0f, 2.0f, 3.0f, 4.0f);
vector char v6 = (vector char)((vector int)(1+2, -2, (int)(2.0 * 3), -(5-3)));
