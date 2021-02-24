// RUN: %clang_cc1 -std=c++14 %s -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// Ensure no warnings are emitted in c++17.
// RUN: %clang_cc1 -std=c++17 %s -verify=cxx17
// RUN: %clang_cc1 -std=c++14 %s -fixit-recompile -fixit-to-temporary -Werror

// cxx17-no-diagnostics

static_assert(true && "String");
// CHECK-DAG: {[[@LINE-1]]:20-[[@LINE-1]]:22}:","

// String literal prefixes are good.
static_assert(true && R"(RawString)");
// CHECK-DAG: {[[@LINE-1]]:20-[[@LINE-1]]:22}:","
static_assert(true && L"RawString");
// CHECK-DAG: {[[@LINE-1]]:20-[[@LINE-1]]:22}:","

static_assert(true);
// CHECK-DAG: {[[@LINE-1]]:19-[[@LINE-1]]:19}:", \"\""

// While its technically possible to transform this to
// static_assert(true, "String") we don't attempt this fix.
static_assert("String" && true);
// CHECK-DAG: {[[@LINE-1]]:31-[[@LINE-1]]:31}:", \"\""

// Don't be smart and look in parentheses.
static_assert((true && "String"));
// CHECK-DAG: {[[@LINE-1]]:33-[[@LINE-1]]:33}:", \"\""
