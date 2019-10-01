//==--- DiagnosticRefactoringKinds.td - refactoring diagnostics -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Refactoring Diagnostics
//===----------------------------------------------------------------------===//

let Component = "Refactoring" in {

let CategoryName = "Refactoring Invocation Issue" in {

def err_refactor_no_selection : Error<"refactoring action can't be initiated "
  "without a selection">;
def err_refactor_selection_no_symbol : Error<"there is no symbol at the given "
  "location">;
def err_refactor_selection_invalid_ast : Error<"the provided selection does "
  "not overlap with the AST nodes of interest">;

def err_refactor_code_outside_of_function : Error<"the selected code is not a "
  "part of a function's / method's body">;
def err_refactor_extract_simple_expression : Error<"the selected expression "
  "is too simple to extract">;
def err_refactor_extract_prohibited_expression : Error<"the selected "
  "expression can't be extracted">;

}

// On github swift-clang only; to be upstreamed:

let CategoryName = "Rename Issue" in {
def err_rename_builtin_function : Error<"%0 is a builtin function that "
  "cannot be renamed">;
def err_rename_sys_header : Error<"%0 cannot be renamed because it is "
  "declared in a system header">;
def err_method_rename_override_sys_framework : Error<"method %0 cannot be "
  "renamed because it overrides a method declared in a system framework">;
def err_rename_external_source_symbol : Error<"%0 is declared in a %1 file; "
  "rename can be initiated in a %1 file only">;
}

let CategoryName = "Refactoring Continuation Issue" in {

def err_ref_continuation_missing_implementation : Error<
  "no %select{implementation file|@implementation declaration}0 for the "
  "selected %select{declaration|@interface}0 %1; please add one and run the "
  "refactoring action again">;

def err_implement_declared_methods_all_implemented : Error<
  "the selected methods are already implemented">;

}

} // end of Refactoring diagnostics
