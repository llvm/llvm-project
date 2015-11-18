//===--- Action.h - Abstract compilation steps ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_ACTION_H
#define LLVM_CLANG_DRIVER_ACTION_H

#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
namespace opt {
  class Arg;
}
}

namespace clang {
namespace driver {

/// Action - Represent an abstract compilation step to perform.
///
/// An action represents an edge in the compilation graph; typically
/// it is a job to transform an input using some tool.
///
/// The current driver is hard wired to expect actions which produce a
/// single primary output, at least in terms of controlling the
/// compilation. Actions can produce auxiliary files, but can only
/// produce a single output to feed into subsequent actions.
class Action {
public:
  typedef ActionList::size_type size_type;
  typedef ActionList::iterator iterator;
  typedef ActionList::const_iterator const_iterator;

  enum ActionClass {
    InputClass = 0,
    BindArchClass,
    CudaDeviceClass,
    CudaHostClass,
    PreprocessJobClass,
    PrecompileJobClass,
    AnalyzeJobClass,
    MigrateJobClass,
    CompileJobClass,
    BackendJobClass,
    AssembleJobClass,
    LinkJobClass,
    LipoJobClass,
    DsymutilJobClass,
    VerifyDebugInfoJobClass,
    VerifyPCHJobClass,

    JobClassFirst=PreprocessJobClass,
    JobClassLast=VerifyPCHJobClass
  };

  static const char *getClassName(ActionClass AC);

private:
  ActionClass Kind;

  /// The output type of this action.
  types::ID Type;

  ActionList Inputs;

  unsigned OwnsInputs : 1;

protected:
  Action(ActionClass Kind, types::ID Type)
    : Kind(Kind), Type(Type), OwnsInputs(true)  {}
  Action(ActionClass Kind, std::unique_ptr<Action> Input, types::ID Type)
      : Kind(Kind), Type(Type), Inputs(1, Input.release()), OwnsInputs(true) {
  }
  Action(ActionClass Kind, std::unique_ptr<Action> Input)
      : Kind(Kind), Type(Input->getType()), Inputs(1, Input.release()),
        OwnsInputs(true) {}
  Action(ActionClass Kind, const ActionList &Inputs, types::ID Type)
    : Kind(Kind), Type(Type), Inputs(Inputs), OwnsInputs(true) {}
public:
  virtual ~Action();

  const char *getClassName() const { return Action::getClassName(getKind()); }

  bool getOwnsInputs() { return OwnsInputs; }
  void setOwnsInputs(bool Value) { OwnsInputs = Value; }

  ActionClass getKind() const { return Kind; }
  types::ID getType() const { return Type; }

  ActionList &getInputs() { return Inputs; }
  const ActionList &getInputs() const { return Inputs; }

  size_type size() const { return Inputs.size(); }

  iterator begin() { return Inputs.begin(); }
  iterator end() { return Inputs.end(); }
  const_iterator begin() const { return Inputs.begin(); }
  const_iterator end() const { return Inputs.end(); }
};

class InputAction : public Action {
  virtual void anchor();
  const llvm::opt::Arg &Input;

public:
  InputAction(const llvm::opt::Arg &Input, types::ID Type);

  const llvm::opt::Arg &getInputArg() const { return Input; }

  static bool classof(const Action *A) {
    return A->getKind() == InputClass;
  }
};

class BindArchAction : public Action {
  virtual void anchor();
  /// The architecture to bind, or 0 if the default architecture
  /// should be bound.
  const char *ArchName;

public:
  BindArchAction(std::unique_ptr<Action> Input, const char *ArchName);

  const char *getArchName() const { return ArchName; }

  static bool classof(const Action *A) {
    return A->getKind() == BindArchClass;
  }
};

class CudaDeviceAction : public Action {
  virtual void anchor();
  /// GPU architecture to bind -- e.g 'sm_35'.
  const char *GpuArchName;
  /// True when action results are not consumed by the host action (e.g when
  /// -fsyntax-only or --cuda-device-only options are used).
  bool AtTopLevel;

public:
  CudaDeviceAction(std::unique_ptr<Action> Input, const char *ArchName,
                   bool AtTopLevel);

  const char *getGpuArchName() const { return GpuArchName; }
  bool isAtTopLevel() const { return AtTopLevel; }

  static bool classof(const Action *A) {
    return A->getKind() == CudaDeviceClass;
  }
};

class CudaHostAction : public Action {
  virtual void anchor();
  ActionList DeviceActions;

public:
  CudaHostAction(std::unique_ptr<Action> Input,
                 const ActionList &DeviceActions);
  ~CudaHostAction() override;

  const ActionList &getDeviceActions() const { return DeviceActions; }

  static bool classof(const Action *A) { return A->getKind() == CudaHostClass; }
};

class JobAction : public Action {
  virtual void anchor();
protected:
  JobAction(ActionClass Kind, std::unique_ptr<Action> Input, types::ID Type);
  JobAction(ActionClass Kind, const ActionList &Inputs, types::ID Type);

public:
  static bool classof(const Action *A) {
    return (A->getKind() >= JobClassFirst &&
            A->getKind() <= JobClassLast);
  }
};

class PreprocessJobAction : public JobAction {
  void anchor() override;
public:
  PreprocessJobAction(std::unique_ptr<Action> Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == PreprocessJobClass;
  }
};

class PrecompileJobAction : public JobAction {
  void anchor() override;
public:
  PrecompileJobAction(std::unique_ptr<Action> Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == PrecompileJobClass;
  }
};

class AnalyzeJobAction : public JobAction {
  void anchor() override;
public:
  AnalyzeJobAction(std::unique_ptr<Action> Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == AnalyzeJobClass;
  }
};

class MigrateJobAction : public JobAction {
  void anchor() override;
public:
  MigrateJobAction(std::unique_ptr<Action> Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == MigrateJobClass;
  }
};

class CompileJobAction : public JobAction {
  void anchor() override;
public:
  CompileJobAction(std::unique_ptr<Action> Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == CompileJobClass;
  }
};

class BackendJobAction : public JobAction {
  void anchor() override;
public:
  BackendJobAction(std::unique_ptr<Action> Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == BackendJobClass;
  }
};

class AssembleJobAction : public JobAction {
  void anchor() override;
public:
  AssembleJobAction(std::unique_ptr<Action> Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == AssembleJobClass;
  }
};

class LinkJobAction : public JobAction {
  void anchor() override;
public:
  LinkJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == LinkJobClass;
  }
};

class LipoJobAction : public JobAction {
  void anchor() override;
public:
  LipoJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == LipoJobClass;
  }
};

class DsymutilJobAction : public JobAction {
  void anchor() override;
public:
  DsymutilJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == DsymutilJobClass;
  }
};

class VerifyJobAction : public JobAction {
  void anchor() override;
public:
  VerifyJobAction(ActionClass Kind, std::unique_ptr<Action> Input,
                  types::ID Type);
  static bool classof(const Action *A) {
    return A->getKind() == VerifyDebugInfoJobClass ||
           A->getKind() == VerifyPCHJobClass;
  }
};

class VerifyDebugInfoJobAction : public VerifyJobAction {
  void anchor() override;
public:
  VerifyDebugInfoJobAction(std::unique_ptr<Action> Input, types::ID Type);
  static bool classof(const Action *A) {
    return A->getKind() == VerifyDebugInfoJobClass;
  }
};

class VerifyPCHJobAction : public VerifyJobAction {
  void anchor() override;
public:
  VerifyPCHJobAction(std::unique_ptr<Action> Input, types::ID Type);
  static bool classof(const Action *A) {
    return A->getKind() == VerifyPCHJobClass;
  }
};

} // end namespace driver
} // end namespace clang

#endif
