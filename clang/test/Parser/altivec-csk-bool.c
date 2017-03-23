// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -target-feature +altivec -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-feature +altivec -fsyntax-only %s

// PR16456: Verify that bool, true, false are treated as context-sensitive
// keywords (and therefore available for use as identifiers) when in
// Altivec mode.

typedef enum {
  false_value = 0,
  true_value = 1
} bool;

#define true true_value
#define false false_value

