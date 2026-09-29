// RUN: %clang_cc1 -triple arm-none-eabi                               -fsyntax-only -verify=default      %s
// RUN: %clang_cc1 -triple arm-none-eabi -target-feature +execute-only -fsyntax-only -verify=execute-only %s

// default-no-diagnostics

/// Thread-local code generation requires constant pools because most of the
/// relocations needed for it operate on data, so it cannot be used with
/// -mexecute-only (or -mpure-code, which is aliased in the driver).

_Thread_local int t; // execute-only-error {{thread-local storage is not supported for the current target}}
