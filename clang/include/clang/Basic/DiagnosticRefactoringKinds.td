//==--- DiagnosticRefactoringKinds.td - refactoring diagnostics -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Refactoring Diagnostics
//===----------------------------------------------------------------------===//

let Component = "Refactoring" in {

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
