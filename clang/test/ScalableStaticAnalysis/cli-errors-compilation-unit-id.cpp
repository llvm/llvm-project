// CLI errors: missing or empty --ssaf-compilation-unit-id= when
// --ssaf-tu-summary-file= is set. Both forms emit the same default-error
// diagnostic and skip TU-summary writing while leaving the rest of the
// compile pipeline alone.

// DEFINE: %{filecheck} = FileCheck %s --match-full-lines --check-prefix

// =============================================================================
// 1. Missing --ssaf-compilation-unit-id= entirely.
// =============================================================================

// RUN: rm -rf %t && mkdir -p %t
// RUN: not %clang     -c %s -o %t/test.o --ssaf-extract-summaries=CallGraph --ssaf-tu-summary-file=%t/tu-absent.json 2>&1 | %{filecheck}=MISSING
// RUN: not %clang_cc1    %s              --ssaf-extract-summaries=CallGraph --ssaf-tu-summary-file=%t/tu-absent.json 2>&1 | %{filecheck}=MISSING
// MISSING: error: option '--ssaf-tu-summary-file=' requires '--ssaf-compilation-unit-id=' to be set [-Wscalable-static-analysis-framework]
// RUN: not test -e %t/tu-absent.json

// =============================================================================
// 2. --ssaf-compilation-unit-id= present with empty value.
// =============================================================================

// RUN: rm -rf %t && mkdir -p %t
// RUN: not %clang     -c %s -o %t/test.o --ssaf-extract-summaries=CallGraph --ssaf-compilation-unit-id= --ssaf-tu-summary-file=%t/tu-empty.json 2>&1 | %{filecheck}=MISSING
// RUN: not %clang_cc1    %s              --ssaf-extract-summaries=CallGraph --ssaf-compilation-unit-id= --ssaf-tu-summary-file=%t/tu-empty.json 2>&1 | %{filecheck}=MISSING
// RUN: not test -e %t/tu-empty.json

// =============================================================================
// 3. The diagnostic is downgradable; under -Wno-error= the compile still
//    produces its normal object output and no TU summary file.
// =============================================================================

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -c %s -o %t/test.o -Wno-error=scalable-static-analysis-framework --ssaf-extract-summaries=CallGraph --ssaf-tu-summary-file=%t/tu-warn.json 2>&1 | %{filecheck}=WARNING
// WARNING: warning: option '--ssaf-tu-summary-file=' requires '--ssaf-compilation-unit-id=' to be set [-Wscalable-static-analysis-framework]
// RUN: test -e %t/test.o
// RUN: not test -e %t/tu-warn.json

// And it can be silenced entirely with -Wno-.
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -c %s -o %t/test.o -Wno-scalable-static-analysis-framework --ssaf-extract-summaries=CallGraph --ssaf-tu-summary-file=%t/tu-silent.json 2>&1 | count 0
// RUN: test -e %t/test.o
// RUN: not test -e %t/tu-silent.json

// =============================================================================
// 4. --ssaf-compilation-unit-id= alone is a no-op (no other --ssaf-* flag).
// =============================================================================

// RUN: %clang     -fsyntax-only %s --ssaf-compilation-unit-id=cu-X 2>&1 | count 0
// RUN: %clang_cc1 -fsyntax-only %s --ssaf-compilation-unit-id=cu-X 2>&1 | count 0

void foo() {}
