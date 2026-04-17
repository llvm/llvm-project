// RUN: %check_clang_tidy %s misc-use-internal-linkage %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     misc-use-internal-linkage.AnalyzeFunctions: false, \
// RUN:     misc-use-internal-linkage.AnalyzeVariables: false, \
// RUN:     misc-use-internal-linkage.AnalyzeTypes: false \
// RUN:   }}"

// CHECK-MESSAGES: warning: the 'misc-use-internal-linkage' check will not perform any analysis because its 'AnalyzeFunctions', 'AnalyzeVariables', and 'AnalyzeTypes' options have all been set to false [clang-tidy-config]
