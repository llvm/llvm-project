// The source-edit-generation diagnostics are downgradable via
// `-Wno-error=scalable-static-analysis-framework` and silenceable via
// `-Wno-scalable-static-analysis-framework`. In both cases the
// compilation continues normally and produces its object file, but no
// edit or report file is written.

// DEFINE: %{filecheck} = FileCheck %s --match-full-lines --check-prefix
// DEFINE: %{options} = --ssaf-source-transformation=does-not-exist \
// DEFINE:   --ssaf-global-scope-analysis-result=%S/Inputs/empty-suite.json \
// DEFINE:   --ssaf-src-edit-file=%t/edits.yaml \
// DEFINE:   --ssaf-transformation-report-file=%t/report.sarif \
// DEFINE:   --ssaf-compilation-unit-id=cu

// =============================================================================
// 1. -Wno-error=scalable-static-analysis-framework downgrades to a warning.
// =============================================================================

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -c %s -o %t/test.o -Wno-error=scalable-static-analysis-framework %{options} 2>&1 | %{filecheck}=WARNING
// WARNING: warning: no source transformation registered with name: does-not-exist [-Wscalable-static-analysis-framework]
// RUN: test -e %t/test.o
// RUN: not test -e %t/edits.yaml
// RUN: not test -e %t/report.sarif

// =============================================================================
// 2. -Wno-scalable-static-analysis-framework silences the diagnostic.
// =============================================================================

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -c %s -o %t/test.o -Wno-scalable-static-analysis-framework %{options} 2>&1 | count 0
// RUN: test -e %t/test.o
// RUN: not test -e %t/edits.yaml
// RUN: not test -e %t/report.sarif

void foo() {}
