// End-to-end test of the source-edit-generation pipeline driven by the
// `test-transformation` plugin. Walks every function in the main source
// file, inserts a zero-length `/*T*/` comment at each function body's
// start, and emits one `test-touches-function` finding per function.
// The finding's level is `Warning` if the function's USR is in the input
// WPASuite, `Note` otherwise — verifying the WPASuite is actually read.

// REQUIRES: plugins

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -load %llvmshlibdir/SSAFTestTransformationPlugin%pluginext \
// RUN:   --ssaf-source-transformation=test-transformation \
// RUN:   --ssaf-global-scope-analysis-result=%S/Inputs/two-function-suite.json \
// RUN:   --ssaf-src-edit-file=%t/edits.yaml \
// RUN:   --ssaf-transformation-report-file=%t/report.sarif \
// RUN:   --ssaf-compilation-unit-id=cu \
// RUN:   -emit-obj -o %t/test.o %s

// RUN: FileCheck --check-prefix=EDITS --input-file=%t/edits.yaml %s
// EDITS:      MainSourceFile: {{.*}}happy-path.cpp
// EDITS:      Replacements:
// EDITS-DAG:    Offset:          {{[0-9]+}}
// EDITS-DAG:    ReplacementText: '/*T*/'

// RUN: FileCheck --check-prefix=REPORT --input-file=%t/report.sarif %s
// REPORT-DAG: "name": "clang-ssaf"
// REPORT-DAG: "fullName": {{.*}}test-transformation
// REPORT-DAG: "ruleId": "test-touches-function"
// REPORT-DAG: "level": "warning"
// REPORT-DAG: "uri": "file://{{.*}}happy-path.cpp"

// `foo` and `bar` are in two-function-suite.json's id_table, so their
// findings escalate from Note to Warning. `baz` is not — its finding
// stays at Note.
void foo() {}
void bar() {}
void baz() {}
