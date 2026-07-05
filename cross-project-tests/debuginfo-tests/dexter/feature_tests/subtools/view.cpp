// Purpose:
//      Check the `view` subtool works with typical inputs.
//
// RUN: rm -rf %t
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-heuristic --binary %t \
// RUN:   --results %t.results -- %s
//
// RUN: %dexter_base view %t.results/view.cpp.dextIR | FileCheck %s
// CHECK: ## BEGIN
// CHECK: ## END

int main() {
    int a = 0;
    return 0; //DexLabel('ret')
}
// DexExpectWatchValue('a', '0', on_line=ref('ret'))
