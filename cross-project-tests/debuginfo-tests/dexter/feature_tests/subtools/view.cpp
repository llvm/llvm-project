// Purpose:
//      Check the `view` subtool works with typical inputs.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t --results %t.results -- %s
//
// RUN: %dexter_base view %t.results/view.cpp.dextIR | FileCheck --dump-input-context=999999999 %s
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
