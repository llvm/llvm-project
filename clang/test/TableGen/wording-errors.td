// RUN: not clang-tblgen -gen-clang-diags-defs -I%S %s -o /dev/null 2>&1 | FileCheck %s
include "DiagnosticBase.inc"

// Ensure we catch a capital letter at the start of a diagnostic.
def zero : Error<
  "This is bad">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not start with a capital letter; 'This' is invalid

// Test that we also correctly handle selections.
def one : Error<
  "%select{|or}0 That">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not start with a capital letter; 'That' is invalid
def two : Error<
  "%select{as does|}0 This">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not start with a capital letter; 'This' is invalid
def three : Error<
  "%select{and||of course}0 Whatever">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not start with a capital letter; 'Whatever' is invalid

// Test that we accept the following cases.
def four : Error<
  "this is fine">;
def five : Error<
  "%select{this|is|also}0 Fine">;
def six : Error<
  "%select{this|is|also|}0 fine">;
def seven : Error<
  "%select{ARC|C|C23|C++14|OpenMP}0 are also fine">;

// Next, test that we catch punctuation at the end of the diagnostic.
def eight : Error<
  "punctuation is bad.">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not end with punctuation; '.' is invalid
def nine : Error<
  "it's really bad!">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not end with punctuation; '!' is invalid
def ten : Error<
  "we also catch %select{punctuation.|in select}0">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not end with punctuation; '.' is invalid
def eleven : Error<
  "and %select{|here.}0">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not end with punctuation; '.' is invalid
def twelve : Error<
  "and %select{here.|}0">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not end with punctuation; '.' is invalid
def thirteen : Error<
  "and even %select{|here.|}0">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not end with punctuation; '.' is invalid
def fourteen : Error<
  "and %select{here}0.">;
// CHECK-DAG: wording-errors.td:[[@LINE-2]]:5: error: Diagnostics should not end with punctuation; '.' is invalid

// Test that we accept the following cases.
def fifteen : Error<
  "question marks are intentionally okay?">;
