// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// Test that 'clang-cl /E' treats inputs as C++ if the extension is
// unrecognized. midl relies on this. See PR40140.

// Use a plain .cpp extension first.
// RUN: %clang_cl /E -- %s | FileCheck %s

// Copy to use .idl as the extension.
// RUN: cp %s %t.idl
// RUN: %clang_cl /E -- %t.idl | FileCheck %s

#ifdef __cplusplus
struct IsCPlusPlus {};
#endif

// CHECK: struct IsCPlusPlus {};
