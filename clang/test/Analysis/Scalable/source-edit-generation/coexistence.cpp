// Stage-1 (TU-summary extraction) and stage-2 (source-edit generation) can
// both be active in a single clang invocation. Their options do not interact
// at the data layer — the source transformation reads its WPASuite from
// disk, not from the in-flight extractor. The two pipelines stack as
// independent ASTConsumers; both produce their per-TU output files.

// REQUIRES: plugins

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -load %llvmshlibdir/SSAFTestTransformationPlugin%pluginext \
// RUN:   --ssaf-extract-summaries=CallGraph \
// RUN:   --ssaf-tu-summary-file=%t/tu.json \
// RUN:   --ssaf-source-transformation=test-transformation \
// RUN:   --ssaf-global-scope-analysis-result=%S/Inputs/empty-suite.json \
// RUN:   --ssaf-src-edit-file=%t/edits.yaml \
// RUN:   --ssaf-transformation-report-file=%t/report.sarif \
// RUN:   --ssaf-compilation-unit-id=cu \
// RUN:   -emit-obj -o %t/test.o %s

// All four artifacts must be present.
// RUN: test -e %t/test.o
// RUN: test -e %t/tu.json
// RUN: test -e %t/edits.yaml
// RUN: test -e %t/report.sarif

// And the source-transformation outputs are non-trivial (the plugin emits
// one replacement and one finding per function in the main file).
// RUN: FileCheck --check-prefix=EDITS --input-file=%t/edits.yaml %s
// EDITS:    ReplacementText: '/*T*/'
// RUN: FileCheck --check-prefix=REPORT --input-file=%t/report.sarif %s
// REPORT:   "ruleId": "test-touches-function"

void foo() {}
