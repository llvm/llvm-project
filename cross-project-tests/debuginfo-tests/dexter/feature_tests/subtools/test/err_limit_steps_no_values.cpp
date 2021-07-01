// Purpose:
//    Check that specifying an expression without any values to compare against
//    in a \DexLimitSteps command results in a useful error message.
//    Use --binary switch to trick dexter into skipping the build step.
//
// REQUIRES: system-linux
// RUN: not %dexter_base test --binary %s --debugger 'lldb' -- %s | FileCheck %s
// CHECK: parser error:{{.*}}err_limit_steps_no_values.cpp(10): expected 0 or at least 2 positional arguments

// DexLimitSteps('test')
