//===-- CommandLineOpts.cpp -- shared command line options ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// This file defines some shared command-line options that can be used when
/// debugging the test tools.

#include "flang/Optimizer/Passes/CommandLineOpts.h"

using namespace llvm;

#define DisableOption(DOName, DOOption, DODescription)                         \
  cl::opt<bool> disable##DOName("disable-" DOOption,                           \
                                cl::desc("disable " DODescription " pass"),    \
                                cl::init(false), cl::Hidden)
#define EnableOption(EOName, EOOption, EODescription)                          \
  cl::opt<bool> enable##EOName("enable-" EOOption,                             \
                               cl::desc("enable " EODescription " pass"),      \
                               cl::init(false), cl::Hidden)

cl::opt<bool> dynamicArrayStackToHeapAllocation(
    "fdynamic-heap-array",
    cl::desc("place all array allocations of dynamic size on the heap"),
    cl::init(false), cl::Hidden);

cl::opt<std::size_t> arrayStackAllocationThreshold(
    "fstack-array-size",
    cl::desc(
        "place all array allocations more than <size> elements on the heap"),
    cl::init(~static_cast<std::size_t>(0)), cl::Hidden);

cl::opt<bool> ignoreMissingTypeDescriptors(
    "ignore-missing-type-desc",
    cl::desc("ignore failures to find derived type descriptors when "
             "translating FIR to LLVM"),
    cl::init(false), cl::Hidden);

OptimizationLevel defaultOptLevel{OptimizationLevel::O0};

codegenoptions::DebugInfoKind noDebugInfo{codegenoptions::NoDebugInfo};

/// Optimizer Passes
DisableOption(CfgConversion, "cfg-conversion", "disable FIR to CFG pass");
DisableOption(FirAvc, "avc", "array value copy analysis and transformation");
DisableOption(FirMao, "memory-allocation-opt",
              "memory allocation optimization");

DisableOption(FirAliasTags, "fir-alias-tags", "fir alias analysis");
cl::opt<bool> useOldAliasTags(
    "use-old-alias-tags",
    cl::desc("Use a single TBAA tree for all functions and do not use "
             "the FIR alias tags pass"),
    cl::init(false), cl::Hidden);

/// CodeGen Passes
DisableOption(CodeGenRewrite, "codegen-rewrite", "rewrite FIR for codegen");
DisableOption(TargetRewrite, "target-rewrite", "rewrite FIR for target");
DisableOption(DebugInfo, "debug-info", "Add debug info");
DisableOption(FirToLlvmIr, "fir-to-llvmir", "FIR to LLVM-IR dialect");
DisableOption(LlvmIrToLlvm, "llvm", "conversion to LLVM");
DisableOption(BoxedProcedureRewrite, "boxed-procedure-rewrite",
              "rewrite boxed procedures");

DisableOption(ExternalNameConversion, "external-name-interop",
              "convert names with external convention");
EnableOption(ConstantArgumentGlobalisation, "constant-argument-globalisation",
             "the local constant argument to global constant conversion");
DisableOption(CompilerGeneratedNamesConversion, "compiler-generated-names",
              "replace special symbols in compiler generated names");
