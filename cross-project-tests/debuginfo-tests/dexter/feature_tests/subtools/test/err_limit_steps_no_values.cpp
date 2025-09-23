// Purpose:
//    Check that specifying an expression without any values to compare against
//    in a \DexLimitSteps command results in a useful error message.
//    Use --binary switch to trick dexter into skipping the build step.
//
// RUN: not %dexter_base test --binary %s %dexter_regression_test_debugger_args -- %s | FileCheck %s
// CHECK: parser error:{{.*}}err_limit_steps_no_values.cpp(9): expected 0 or at least 2 positional arguments

// DexLimitSteps('test')
