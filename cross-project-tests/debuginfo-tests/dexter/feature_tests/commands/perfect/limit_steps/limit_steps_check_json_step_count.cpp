// Purpose:
//      Check number of step lines are correctly reported in json output.
//
// XFAIL: system-darwin
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t --verbose -- %s | FileCheck %s
// CHECK: limit_steps_check_json_step_count.cpp
// CHECK: ## BEGIN ##
// CHECK-COUNT-3: json_step_count.cpp",

int main() {
  int result = 0;
  for(int ix = 0; ix != 10; ++ix) {
    int index = ix;
    result += index; // DexLabel('check')
  }
}

// DexExpectWatchValue('index', 2, 7, 9, on_line=ref('check'))
// DexLimitSteps('ix', 2, 7, 9, on_line=ref('check'))
