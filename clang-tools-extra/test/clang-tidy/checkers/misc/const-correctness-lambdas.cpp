// RUN: %check_clang_tidy %s misc-const-correctness %t \
// RUN: -config='{CheckOptions: \
// RUN:  {misc-const-correctness.AnalyzeAutoVariables: true,\
// RUN:   misc-const-correctness.AnalyzeValues: true,\
// RUN:   misc-const-correctness.AnalyzeLambdas: false}}' \
// RUN: -- -fno-delayed-template-parsing

void lambdas_ignored_but_other_auto_variables_diagnosed() {
  auto i = 0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'i' of type 'int' can be declared 'const'
  // CHECK-FIXES: auto const i = 0;

  auto lambda = [] {};
}
