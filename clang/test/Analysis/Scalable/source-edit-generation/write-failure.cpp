// When the source-edit or transformation-report writer's `write` returns an
// `llvm::Error`, the framework reports a default-error diagnostic. With
// `-Wno-error=scalable-static-analysis-framework` the diagnostic downgrades
// to a warning and the rest of the compile pipeline finishes normally.

// REQUIRES: plugins

// RUN: rm -rf %t && mkdir -p %t

// =============================================================================
// 1. Source-edit write fails because the parent directory does not exist.
// =============================================================================

// RUN: %clang_cc1 -load %llvmshlibdir/SSAFTestTransformationPlugin%pluginext \
// RUN:   -Wno-error=scalable-static-analysis-framework \
// RUN:   --ssaf-source-transformation=test-transformation \
// RUN:   --ssaf-global-scope-analysis-result=%S/Inputs/empty-suite.json \
// RUN:   --ssaf-src-edit-file=%t/missing-dir/edits.yaml \
// RUN:   --ssaf-transformation-report-file=%t/report.sarif \
// RUN:   --ssaf-compilation-unit-id=cu \
// RUN:   -emit-obj -o %t/test.o %s 2>&1 | FileCheck --check-prefix=EDIT-FAIL %s
// EDIT-FAIL: warning: failed to write source edits to '{{.*}}/missing-dir/edits.yaml'{{.*}}[-Wscalable-static-analysis-framework]
// RUN: test -e %t/test.o

// =============================================================================
// 2. Transformation-report write fails.
// =============================================================================

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -load %llvmshlibdir/SSAFTestTransformationPlugin%pluginext \
// RUN:   -Wno-error=scalable-static-analysis-framework \
// RUN:   --ssaf-source-transformation=test-transformation \
// RUN:   --ssaf-global-scope-analysis-result=%S/Inputs/empty-suite.json \
// RUN:   --ssaf-src-edit-file=%t/edits.yaml \
// RUN:   --ssaf-transformation-report-file=%t/missing-dir/report.sarif \
// RUN:   --ssaf-compilation-unit-id=cu \
// RUN:   -emit-obj -o %t/test.o %s 2>&1 | FileCheck --check-prefix=REPORT-FAIL %s
// REPORT-FAIL: warning: failed to write transformation report to '{{.*}}/missing-dir/report.sarif'{{.*}}[-Wscalable-static-analysis-framework]
// RUN: test -e %t/test.o

void foo() {}
