// RUN: %clang_cc1 -E -verify -o - %s | FileCheck %s
// expected-no-diagnostics

<#placeholder#>; // CHECK: <#placeholder#>;
