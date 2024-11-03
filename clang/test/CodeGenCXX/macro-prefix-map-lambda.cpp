// RUN: %clang_cc1 -triple %itanium_abi_triple -fmacro-prefix-map=%p=./UNLIKELY_PATH/empty %s -emit-llvm -o - | FileCheck %s

template<typename f>
auto lambdatest(f&& cb) {
  const char *s = __PRETTY_FUNCTION__;
  return s;
}

int main() {
  auto *s = lambdatest([](){});
// CHECK: @"__PRETTY_FUNCTION__._Z10lambdatestIZ4mainE3$_0EDaOT_" = private unnamed_addr constant [{{[0-9]+}} x i8] c"auto lambdatest(f &&) [f = (lambda at ./UNLIKELY_PATH/empty{{/|\\\\}}{{.*}}.cpp:[[#@LINE-1]]:24)]\00", align {{[0-9]+}}

  return 0;
}
