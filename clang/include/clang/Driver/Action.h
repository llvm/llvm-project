//===--- Action.h - Abstract compilation steps ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_ACTION_H_
#define CLANG_DRIVER_ACTION_H_

#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace driver {
  class Arg;

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
    PreprocessJobClass,
    PrecompileJobClass,
    AnalyzeJobClass,
    MigrateJobClass,
    CompileJobClass,
    AssembleJobClass,
    LinkJobClass,
    LipoJobClass,
    DsymutilJobClass,
    VerifyJobClass,

    JobClassFirst=PreprocessJobClass,
    JobClassLast=VerifyJobClass
  };

  static const char *getClassName(ActionClass AC);

private:
  ActionClass Kind;

  /// The output type of this action.
  types::ID Type;

  ActionList Inputs;

  unsigned OwnsInputs : 1;

protected:
  Action(ActionClass _Kind, types::ID _Type)
    : Kind(_Kind), Type(_Type), OwnsInputs(true)  {}
  Action(ActionClass _Kind, Action *Input, types::ID _Type)
    : Kind(_Kind), Type(_Type), Inputs(&Input, &Input + 1), OwnsInputs(true) {}
  Action(ActionClass _Kind, const ActionList &_Inputs, types::ID _Type)
    : Kind(_Kind), Type(_Type), Inputs(_Inputs), OwnsInputs(true) {}
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
  const Arg &Input;
public:
  InputAction(const Arg &_Input, types::ID _Type);

  const Arg &getInputArg() const { return Input; }

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
  BindArchAction(Action *Input, const char *_ArchName);

  const char *getArchName() const { return ArchName; }

  static bool classof(const Action *A) {
    return A->getKind() == BindArchClass;
  }
};

class JobAction : public Action {
  virtual void anchor();
protected:
  JobAction(ActionClass Kind, Action *Input, types::ID Type);
  JobAction(ActionClass Kind, const ActionList &Inputs, types::ID Type);

public:
  static bool classof(const Action *A) {
    return (A->getKind() >= JobClassFirst &&
            A->getKind() <= JobClassLast);
  }
};

class PreprocessJobAction : public JobAction {
  virtual void anchor();
public:
  PreprocessJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == PreprocessJobClass;
  }
};

class PrecompileJobAction : public JobAction {
  virtual void anchor();
public:
  PrecompileJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == PrecompileJobClass;
  }
};

class AnalyzeJobAction : public JobAction {
  virtual void anchor();
public:
  AnalyzeJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == AnalyzeJobClass;
  }
};

class MigrateJobAction : public JobAction {
  virtual void anchor();
public:
  MigrateJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == MigrateJobClass;
  }
};

class CompileJobAction : public JobAction {
  virtual void anchor();
public:
  CompileJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == CompileJobClass;
  }
};

class AssembleJobAction : public JobAction {
  virtual void anchor();
public:
  AssembleJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == AssembleJobClass;
  }
};

class LinkJobAction : public JobAction {
  virtual void anchor();
public:
  LinkJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == LinkJobClass;
  }
};

class LipoJobAction : public JobAction {
  virtual void anchor();
public:
  LipoJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == LipoJobClass;
  }
};

class DsymutilJobAction : public JobAction {
  virtual void anchor();
public:
  DsymutilJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == DsymutilJobClass;
  }
};

class VerifyJobAction : public JobAction {
  virtual void anchor();
public:
  VerifyJobAction(ActionList &Inputs, types::ID Type);
  static bool classof(const Action *A) {
    return A->getKind() == VerifyJobClass;
  }
};

} // end namespace driver
} // end namespace clang

#endif
