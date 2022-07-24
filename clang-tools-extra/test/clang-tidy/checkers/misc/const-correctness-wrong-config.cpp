// RUN: %check_clang_tidy %s misc-const-correctness %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: "misc-const-correctness.AnalyzeValues", value: false},\
// RUN:   {key: "misc-const-correctness.AnalyzeReferences", value: false},\
// RUN:  ]}' -- -fno-delayed-template-parsing

// CHECK-MESSAGES: warning: The check 'misc-const-correctness' will not perform any analysis because both 'AnalyzeValues' and 'AnalyzeReferences' are false. [clang-tidy-config]

void g() {
  int p_local0 = 42;
  // CHECK-FIXES-NOT: int const p_local0 = 42;
}
