// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c23 -MT %s.o --embed-dir=%S/Inputs -dependency-file - | FileCheck %s

// Yes this looks very strange indeed, but the goal is to test that we add
// files referenced by both __has_embed and #embed when we generate
// dependencies, so we're trying to see that both of these files are in the
// output.
#if __has_embed(<jk.txt>)
const char data =
#embed "Inputs/single_byte.txt"
;
_Static_assert('b' == data);
#else
#error "oops"
#endif
// expected-no-diagnostics

// CHECK: embed_dependencies.c.o
// CHECK-NEXT: embed_dependencies.c
// CHECK-NEXT: jk.txt
// CHECK-NEXT: Inputs{{[/\\]}}single_byte.txt

