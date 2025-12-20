// RUN: %check_clang_tidy %s misc-const-correctness %t \
// RUN: -config='{CheckOptions: \
// RUN:  {misc-const-correctness.AnalyzeAutoVariables: false,\
// RUN:   misc-const-correctness.AnalyzeValues: true,\
// RUN:   misc-const-correctness.AnalyzeLambdas: true,\
// RUN:   misc-const-correctness.WarnPointersAsValues: true,\
// RUN:   misc-const-correctness.WarnPointersAsPointers: true,\
// RUN:   misc-const-correctness.TransformPointersAsValues: true}}' \
// RUN: -- -fno-delayed-template-parsing

void auto_types_ignored() {
  int i = 0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'i' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const i = 0;

  auto auto_i = 0;
  auto& auto_ref = auto_i;
  auto auto_lambda = [] {};
  auto *auto_ptr = nullptr;
}
