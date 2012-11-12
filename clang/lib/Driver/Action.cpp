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
  case PreprocessJobClass: return "preprocessor";
  case PrecompileJobClass: return "precompiler";
  case AnalyzeJobClass: return "analyzer";
  case MigrateJobClass: return "migrator";
  case CompileJobClass: return "compiler";
  case AssembleJobClass: return "assembler";
  case LinkJobClass: return "linker";
  case LipoJobClass: return "lipo";
  case DsymutilJobClass: return "dsymutil";
  case VerifyJobClass: return "verify";
  }

  llvm_unreachable("invalid class");
}

void InputAction::anchor() {}

InputAction::InputAction(const Arg &_Input, types::ID _Type)
  : Action(InputClass, _Type), Input(_Input) {
}

void BindArchAction::anchor() {}

BindArchAction::BindArchAction(Action *Input, const char *_ArchName)
  : Action(BindArchClass, Input, Input->getType()), ArchName(_ArchName) {
}

void JobAction::anchor() {}

JobAction::JobAction(ActionClass Kind, Action *Input, types::ID Type)
  : Action(Kind, Input, Type) {
}

JobAction::JobAction(ActionClass Kind, const ActionList &Inputs, types::ID Type)
  : Action(Kind, Inputs, Type) {
}

void PreprocessJobAction::anchor() {}

PreprocessJobAction::PreprocessJobAction(Action *Input, types::ID OutputType)
  : JobAction(PreprocessJobClass, Input, OutputType) {
}

void PrecompileJobAction::anchor() {}

PrecompileJobAction::PrecompileJobAction(Action *Input, types::ID OutputType)
  : JobAction(PrecompileJobClass, Input, OutputType) {
}

void AnalyzeJobAction::anchor() {}

AnalyzeJobAction::AnalyzeJobAction(Action *Input, types::ID OutputType)
  : JobAction(AnalyzeJobClass, Input, OutputType) {
}

void MigrateJobAction::anchor() {}

MigrateJobAction::MigrateJobAction(Action *Input, types::ID OutputType)
  : JobAction(MigrateJobClass, Input, OutputType) {
}

void CompileJobAction::anchor() {}

CompileJobAction::CompileJobAction(Action *Input, types::ID OutputType)
  : JobAction(CompileJobClass, Input, OutputType) {
}

void AssembleJobAction::anchor() {}

AssembleJobAction::AssembleJobAction(Action *Input, types::ID OutputType)
  : JobAction(AssembleJobClass, Input, OutputType) {
}

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

VerifyJobAction::VerifyJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(VerifyJobClass, Inputs, Type) {
}
