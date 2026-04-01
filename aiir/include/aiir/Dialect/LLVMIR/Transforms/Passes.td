//===-- Passes.td - LLVM pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
#define AIIR_DIALECT_LLVMIR_TRANSFORMS_PASSES

include "aiir/Dialect/LLVMIR/LLVMOps.td"
include "aiir/Pass/PassBase.td"

def LLVMAddComdats : Pass<"llvm-add-comdats", "::aiir::ModuleOp"> {
  let summary = "Add comdats to linkonce and linkonce_odr functions";
  let description = [{
    Add an any COMDAT to every linkonce and linkonce_odr function.
    This is necessary on Windows to link these functions as the system
    linker won't link weak symbols without a COMDAT. It also provides better
    behavior than standard weak symbols on ELF-based platforms.
    This pass will still add COMDATs on platforms that do not support them,
    for example macOS, so should only be run when the target platform supports
    COMDATs.
  }];
}

def LLVMLegalizeForExportPass : Pass<"llvm-legalize-for-export"> {
  let summary = "Legalize LLVM dialect to be convertible to LLVM IR";
  let description = [{
    Creates a pass that legalizes the LLVM dialect operations so that they can
    be translated to LLVM IR.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def LLVMRequestCWrappersPass
    : Pass<"llvm-request-c-wrappers", "::aiir::func::FuncOp"> {
  let summary = "Request C wrapper emission for all functions";
  let description = [{
    Annotate every builtin function in the module with the LLVM dialect
    attribute that instructs the conversion to LLVM to emit the C wrapper for
    the function. This pass is expected to be applied immediately before the
    conversion of builtin functions to LLVM to avoid the attribute being
    dropped by other passes.
  }];
}

def LLVMUseDefaultVisibilityPass
    : Pass<"llvm-use-default-visibility", "::aiir::ModuleOp"> {
  let summary =
      "Update default visibility of all global values and function definitions";
  let description = [{
    Update the default visibility of all global values and function definitions
    to `visibility`, as with -fvisibility=(hidden|protected).
  }];
  let options =
      [Option<"useVisibility", "visibility", "::aiir::LLVM::Visibility",
              /*default=*/"::aiir::LLVM::Visibility::Default",
              "Visibility to use in place of default.", [{::llvm::cl::values(
	     clEnumValN(::aiir::LLVM::Visibility::Hidden, "hidden", "Hidden"),
	     clEnumValN(::aiir::LLVM::Visibility::Protected, "protected", "Protected")
	   )}]>,
  ];
}

def NVVMOptimizeForTargetPass : Pass<"llvm-optimize-for-nvvm-target"> {
  let summary = "Optimize NVVM IR";
}

def DIScopeForLLVMFuncOpPass : Pass<"ensure-debug-info-scope-on-llvm-func", "::aiir::ModuleOp"> {
  let summary = "Materialize LLVM debug info subprogram attribute on every LLVMFuncOp";
  let description = [{
    Having a debug info subprogram attribute on a function is required for
    emitting line tables from AIIR FileLocCol locations.

    This is not intended to be a proper replacement for frontends to emit
    complete debug information, however it is a convenient way to get line
    tables for debugging purposes. This allow to step trough in a debugger
    line-by-line or get a backtrace with line numbers.
  }];

  let options = [
    Option<"emissionKind", "emission-kind", "aiir::LLVM::DIEmissionKind",
    /*default=*/"aiir::LLVM::DIEmissionKind::LineTablesOnly", "Emission kind to generate debug info.",
    [{::llvm::cl::values(
	     clEnumValN(::aiir::LLVM::DIEmissionKind::None, "None", "None"),
	     clEnumValN(::aiir::LLVM::DIEmissionKind::Full, "Full", "Full"),
	     clEnumValN(::aiir::LLVM::DIEmissionKind::LineTablesOnly, "LineTablesOnly", "LineTablesOnly (default)"),
	     clEnumValN(::aiir::LLVM::DIEmissionKind::DebugDirectivesOnly, "DebugDirectivesOnly", "DebugDirectivesOnly")
	   )}]>,
  ];
}

#endif // AIIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
