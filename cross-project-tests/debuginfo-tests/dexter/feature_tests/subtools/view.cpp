// Purpose:
//      Check the `view` subtool works with typical inputs.
//
// RUN: %dexter_regression_test --results %t -- %s
//
// RUN: %dexter_base view %t/view.cpp.dextIR | FileCheck %s
// CHECK: ## BEGIN
// CHECK: ## END
//
// # [TODO] This doesn't run if FileCheck fails!
// RUN: rm -rf %t

int main() {
    int a = 0;
    return 0; //DexLabel('ret')
}
// DexExpectWatchValue('a', '0', on_line=ref('ret'))
