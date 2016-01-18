//===--- Action.cpp - Abstract compilation steps --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Action.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
using namespace clang::driver;
using namespace llvm::opt;

Action::~Action() {
  if (OwnsInputs) {
    for (iterator it = begin(), ie = end(); it != ie; ++it)
      delete *it;
  }
}

const char *Action::getClassName(ActionClass AC) {
  switch (AC) {
  case InputClass: return "input";
  case BindArchClass: return "bind-arch";
  case CudaDeviceClass: return "cuda-device";
  case CudaHostClass: return "cuda-host";
  case PreprocessJobClass: return "preprocessor";
  case PrecompileJobClass: return "precompiler";
  case AnalyzeJobClass: return "analyzer";
  case MigrateJobClass: return "migrator";
  case CompileJobClass: return "compiler";
  case BackendJobClass: return "backend";
  case AssembleJobClass: return "assembler";
  case LinkJobClass: return "linker";
  case LipoJobClass: return "lipo";
  case DsymutilJobClass: return "dsymutil";
  case VerifyDebugInfoJobClass: return "verify-debug-info";
  case VerifyPCHJobClass: return "verify-pch";
  }

  llvm_unreachable("invalid class");
}

void InputAction::anchor() {}

InputAction::InputAction(const Arg &_Input, types::ID _Type)
  : Action(InputClass, _Type), Input(_Input) {
}

void BindArchAction::anchor() {}

BindArchAction::BindArchAction(std::unique_ptr<Action> Input,
                               const char *_ArchName)
    : Action(BindArchClass, std::move(Input)), ArchName(_ArchName) {}

void CudaDeviceAction::anchor() {}

CudaDeviceAction::CudaDeviceAction(std::unique_ptr<Action> Input,
                                   const char *ArchName,
                                   const char *DeviceTriple, bool AtTopLevel)
    : Action(CudaDeviceClass, std::move(Input)), GpuArchName(ArchName),
      DeviceTriple(DeviceTriple), AtTopLevel(AtTopLevel) {}

void CudaHostAction::anchor() {}

CudaHostAction::CudaHostAction(std::unique_ptr<Action> Input,
                               const ActionList &DeviceActions,
                               const char *DeviceTriple)
    : Action(CudaHostClass, std::move(Input)), DeviceActions(DeviceActions),
      DeviceTriple(DeviceTriple) {}

CudaHostAction::~CudaHostAction() {
  for (auto &DA : DeviceActions)
    delete DA;
}

void JobAction::anchor() {}

JobAction::JobAction(ActionClass Kind, std::unique_ptr<Action> Input,
                     types::ID Type)
    : Action(Kind, std::move(Input), Type) {}

JobAction::JobAction(ActionClass Kind, const ActionList &Inputs, types::ID Type)
  : Action(Kind, Inputs, Type) {
}

void PreprocessJobAction::anchor() {}

PreprocessJobAction::PreprocessJobAction(std::unique_ptr<Action> Input,
                                         types::ID OutputType)
    : JobAction(PreprocessJobClass, std::move(Input), OutputType) {}

void PrecompileJobAction::anchor() {}

PrecompileJobAction::PrecompileJobAction(std::unique_ptr<Action> Input,
                                         types::ID OutputType)
    : JobAction(PrecompileJobClass, std::move(Input), OutputType) {}

void AnalyzeJobAction::anchor() {}

AnalyzeJobAction::AnalyzeJobAction(std::unique_ptr<Action> Input,
                                   types::ID OutputType)
    : JobAction(AnalyzeJobClass, std::move(Input), OutputType) {}

void MigrateJobAction::anchor() {}

MigrateJobAction::MigrateJobAction(std::unique_ptr<Action> Input,
                                   types::ID OutputType)
    : JobAction(MigrateJobClass, std::move(Input), OutputType) {}

void CompileJobAction::anchor() {}

CompileJobAction::CompileJobAction(std::unique_ptr<Action> Input,
                                   types::ID OutputType)
    : JobAction(CompileJobClass, std::move(Input), OutputType) {}

void BackendJobAction::anchor() {}

BackendJobAction::BackendJobAction(std::unique_ptr<Action> Input,
                                   types::ID OutputType)
    : JobAction(BackendJobClass, std::move(Input), OutputType) {}

void AssembleJobAction::anchor() {}

AssembleJobAction::AssembleJobAction(std::unique_ptr<Action> Input,
                                     types::ID OutputType)
    : JobAction(AssembleJobClass, std::move(Input), OutputType) {}

void LinkJobAction::anchor() {}

LinkJobAction::LinkJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(LinkJobClass, Inputs, Type) {
}

void LipoJobAction::anchor() {}

LipoJobAction::LipoJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(LipoJobClass, Inputs, Type) {
}

void DsymutilJobAction::anchor() {}

DsymutilJobAction::DsymutilJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(DsymutilJobClass, Inputs, Type) {
}

void VerifyJobAction::anchor() {}

VerifyJobAction::VerifyJobAction(ActionClass Kind,
                                 std::unique_ptr<Action> Input, types::ID Type)
    : JobAction(Kind, std::move(Input), Type) {
  assert((Kind == VerifyDebugInfoJobClass || Kind == VerifyPCHJobClass) &&
         "ActionClass is not a valid VerifyJobAction");
}

void VerifyDebugInfoJobAction::anchor() {}

VerifyDebugInfoJobAction::VerifyDebugInfoJobAction(
    std::unique_ptr<Action> Input, types::ID Type)
    : VerifyJobAction(VerifyDebugInfoJobClass, std::move(Input), Type) {}

void VerifyPCHJobAction::anchor() {}

VerifyPCHJobAction::VerifyPCHJobAction(std::unique_ptr<Action> Input,
                                       types::ID Type)
    : VerifyJobAction(VerifyPCHJobClass, std::move(Input), Type) {}
