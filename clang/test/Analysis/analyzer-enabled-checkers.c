// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 %s -o /dev/null -analyzer-checker=core -analyzer-list-enabled-checkers > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// CHECK: OVERVIEW: Clang Static Analyzer Enabled Checkers List
// CHECK: core.CallAndMessage
// CHECK: core.DivideZero
// CHECK: core.DynamicTypePropagation
// CHECK: core.NonNullParamChecker
// CHECK: core.NullDereference
// CHECK: core.StackAddressEscape
// CHECK: core.UndefinedBinaryOperatorResult
// CHECK: core.VLASize
// CHECK: core.builtin.BuiltinFunctions
// CHECK: core.builtin.NoReturnFunctions
// CHECK: core.uninitialized.ArraySubscript
// CHECK: core.uninitialized.Assign
// CHECK: core.uninitialized.Branch
// CHECK: core.uninitialized.CapturedBlockVariable
// CHECK: core.uninitialized.UndefReturn

