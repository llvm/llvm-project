// The flow-sensitive nullability LangOpts (FlowSensitiveNullability,
// NullabilityDefault) are declared NotCompatible in LangOptions.def, so a PCH
// built with them must be rejected when consumed with mismatching values.
//
// checkLangOptions early-returns on the first mismatch, and NullabilityDefault
// is checked before FlowSensitiveNullability, so each case below changes ONLY
// the opt it means to exercise (keeping the other matching) to pin down which
// diagnostic fires.

// Build the PCH with -fflow-sensitive-nullability -fnullability-default=nullable.
// RUN: %clang_cc1 -x c++-header -emit-pch -fflow-sensitive-nullability -fnullability-default=nullable -o %t %S/flow-nullability.h

// (1) Boolean opt mismatch: keep the default matching (nullable) but drop
//     -fflow-sensitive-nullability, so only FlowSensitiveNullability differs.
// RUN: not %clang_cc1 -fsyntax-only -fnullability-default=nullable -include-pch %t %s 2>&1 | FileCheck --check-prefix=NOFLAG %s

// NOFLAG: error: enable flow-sensitive nullability analysis was enabled in precompiled file '{{.*}}' but is currently disabled

// (2) Enum opt mismatch: keep the boolean on but change the default
//     (nonnull vs nullable), so only NullabilityDefault differs.
// RUN: not %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nonnull -include-pch %t %s 2>&1 | FileCheck --check-prefix=DEFAULT %s

// DEFAULT: error: default nullability for unannotated pointers differs in precompiled file '{{.*}}' vs. current file

int unused;
