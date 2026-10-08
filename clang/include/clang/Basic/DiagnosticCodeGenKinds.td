//==--- DiagnosticCodeGenKinds.td - codegen diagnostics -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostics emitted while lowering the AST to code. This component is shared
// between the LLVM IR code generator (clang/lib/CodeGen) and the ClangIR (CIR)
// pipeline (clang/lib/CIR). Existing LLVM IR codegen diagnostics still live in
// DiagnosticFrontendKinds.td and can migrate here incrementally.
//
//===----------------------------------------------------------------------===//

let Component = "CodeGen" in {

// ClangIR (CIR) diagnostics.

def err_cir_to_cir_transform_failed : Error<
    "CIR-to-CIR transformation failed">, DefaultFatal;

def err_cir_verification_failed_pre_passes : Error<
    "CIR module verification error before running CIR-to-CIR passes">,
    DefaultFatal;

// Diagnostics relayed from MLIR-side passes/verifier through the
// clang::DiagnosticsEngine. The full message text comes from MLIR.
def err_cir_mlir_diagnostic    : Error<"%0">;
def warn_cir_mlir_diagnostic   : Warning<"%0">, InGroup<ClangIR>;
def remark_cir_mlir_diagnostic : Remark<"%0">, InGroup<RemarkClangIR>;
def note_cir_mlir_diagnostic   : Note<"%0">;

}
