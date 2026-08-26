// CLI errors for the source-edit-generation pipeline. Every misuse of the
// four `--ssaf-{source-transformation,global-scope-analysis-result,
// src-edit-file,transformation-report-file}=` options emits a default-error
// diagnostic under `-Wscalable-static-analysis-framework`. The runner
// produces no edit/report files and the rest of the compile pipeline is
// untouched.

// DEFINE: %{filecheck} = FileCheck %s --match-full-lines --check-prefix
// DEFINE: %{base} = --ssaf-source-transformation=does-not-exist \
// DEFINE:   --ssaf-global-scope-analysis-result=%S/Inputs/empty-suite.json \
// DEFINE:   --ssaf-src-edit-file=%t/edits.yaml \
// DEFINE:   --ssaf-transformation-report-file=%t/report.sarif \
// DEFINE:   --ssaf-compilation-unit-id=cu

// =============================================================================
// 1. Unknown transformation name.
// =============================================================================

// RUN: rm -rf %t && mkdir -p %t
// RUN: not %clang     -c %s -o %t/test.o %{base} 2>&1 | %{filecheck}=UNKNOWN-NAME
// RUN: not %clang_cc1    %s              %{base} 2>&1 | %{filecheck}=UNKNOWN-NAME
// UNKNOWN-NAME: error: no source transformation registered with name: does-not-exist [-Wscalable-static-analysis-framework]
// RUN: not test -e %t/edits.yaml
// RUN: not test -e %t/report.sarif

// =============================================================================
// 2. Orphan companion options: --ssaf-source-transformation= alone.
// =============================================================================

// RUN: rm -rf %t && mkdir -p %t
// RUN: not %clang -c %s -o %t/test.o --ssaf-source-transformation=does-not-exist 2>&1 | %{filecheck}=ORPHAN-COMPANIONS
// ORPHAN-COMPANIONS-DAG: error: option '--ssaf-source-transformation=' requires '--ssaf-global-scope-analysis-result=' to be set [-Wscalable-static-analysis-framework]
// ORPHAN-COMPANIONS-DAG: error: option '--ssaf-source-transformation=' requires '--ssaf-src-edit-file=' to be set [-Wscalable-static-analysis-framework]
// ORPHAN-COMPANIONS-DAG: error: option '--ssaf-source-transformation=' requires '--ssaf-transformation-report-file=' to be set [-Wscalable-static-analysis-framework]
// ORPHAN-COMPANIONS-DAG: error: option '--ssaf-source-transformation=' requires '--ssaf-compilation-unit-id=' to be set [-Wscalable-static-analysis-framework]

// =============================================================================
// 3. Reverse orphans: edit/report file set without transformation option.
// =============================================================================

// RUN: rm -rf %t && mkdir -p %t
// RUN: not %clang -c %s -o %t/test.o --ssaf-src-edit-file=%t/e.yaml 2>&1 | %{filecheck}=ORPHAN-EDIT
// ORPHAN-EDIT: error: option '--ssaf-src-edit-file=' is ignored without '--ssaf-source-transformation=' [-Wscalable-static-analysis-framework]
// RUN: not test -e %t/e.yaml

// RUN: rm -rf %t && mkdir -p %t
// RUN: not %clang -c %s -o %t/test.o --ssaf-transformation-report-file=%t/r.sarif 2>&1 | %{filecheck}=ORPHAN-REPORT
// ORPHAN-REPORT: error: option '--ssaf-transformation-report-file=' is ignored without '--ssaf-source-transformation=' [-Wscalable-static-analysis-framework]
// RUN: not test -e %t/r.sarif

void foo() {}
