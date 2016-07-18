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

#include "clang/Basic/Cuda.h"
#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class StringRef;

namespace opt {
  class Arg;
}
}

namespace clang {
namespace driver {

class ToolChain;

/// Action - Represent an abstract compilation step to perform.
///
/// An action represents an edge in the compilation graph; typically
/// it is a job to transform an input using some tool.
///
/// The current driver is hard wired to expect actions which produce a
/// single primary output, at least in terms of controlling the
/// compilation. Actions can produce auxiliary files, but can only
/// produce a single output to feed into subsequent actions.
///
/// Actions are usually owned by a Compilation, which creates new
/// actions via MakeAction().
class Action {
public:
  typedef ActionList::size_type size_type;
  typedef ActionList::iterator input_iterator;
  typedef ActionList::const_iterator input_const_iterator;
  typedef llvm::iterator_range<input_iterator> input_range;
  typedef llvm::iterator_range<input_const_iterator> input_const_range;

  enum ActionClass {
    InputClass = 0,
    BindArchClass,
    OffloadClass,
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

    JobClassFirst = PreprocessJobClass,
    JobClassLast = VerifyPCHJobClass
  };

  // The offloading kind determines if this action is binded to a particular
  // programming model. Each entry reserves one bit. We also have a special kind
  // to designate the host offloading tool chain.
  enum OffloadKind {
    OFK_None = 0x00,
    // The host offloading tool chain.
    OFK_Host = 0x01,
    // The device offloading tool chains - one bit for each programming model.
    OFK_Cuda = 0x02,
  };

  static const char *getClassName(ActionClass AC);

private:
  ActionClass Kind;

  /// The output type of this action.
  types::ID Type;

  ActionList Inputs;

protected:
  ///
  /// Offload information.
  ///

  /// The host offloading kind - a combination of kinds encoded in a mask.
  /// Multiple programming models may be supported simultaneously by the same
  /// host.
  unsigned ActiveOffloadKindMask = 0u;
  /// Offloading kind of the device.
  OffloadKind OffloadingDeviceKind = OFK_None;
  /// The Offloading architecture associated with this action.
  const char *OffloadingArch = nullptr;

  Action(ActionClass Kind, types::ID Type) : Action(Kind, ActionList(), Type) {}
  Action(ActionClass Kind, Action *Input, types::ID Type)
      : Action(Kind, ActionList({Input}), Type) {}
  Action(ActionClass Kind, Action *Input)
      : Action(Kind, ActionList({Input}), Input->getType()) {}
  Action(ActionClass Kind, const ActionList &Inputs, types::ID Type)
      : Kind(Kind), Type(Type), Inputs(Inputs) {}

public:
  virtual ~Action();

  const char *getClassName() const { return Action::getClassName(getKind()); }

  ActionClass getKind() const { return Kind; }
  types::ID getType() const { return Type; }

  ActionList &getInputs() { return Inputs; }
  const ActionList &getInputs() const { return Inputs; }

  size_type size() const { return Inputs.size(); }

  input_iterator input_begin() { return Inputs.begin(); }
  input_iterator input_end() { return Inputs.end(); }
  input_range inputs() { return input_range(input_begin(), input_end()); }
  input_const_iterator input_begin() const { return Inputs.begin(); }
  input_const_iterator input_end() const { return Inputs.end(); }
  input_const_range inputs() const {
    return input_const_range(input_begin(), input_end());
  }

  /// Return a string containing the offload kind of the action.
  std::string getOffloadingKindPrefix() const;
  /// Return a string that can be used as prefix in order to generate unique
  /// files for each offloading kind.
  std::string
  getOffloadingFileNamePrefix(llvm::StringRef NormalizedTriple) const;

  /// Set the device offload info of this action and propagate it to its
  /// dependences.
  void propagateDeviceOffloadInfo(OffloadKind OKind, const char *OArch);
  /// Append the host offload info of this action and propagate it to its
  /// dependences.
  void propagateHostOffloadInfo(unsigned OKinds, const char *OArch);
  /// Set the offload info of this action to be the same as the provided action,
  /// and propagate it to its dependences.
  void propagateOffloadInfo(const Action *A);

  unsigned getOffloadingHostActiveKinds() const {
    return ActiveOffloadKindMask;
  }
  OffloadKind getOffloadingDeviceKind() const { return OffloadingDeviceKind; }
  const char *getOffloadingArch() const { return OffloadingArch; }

  /// Check if this action have any offload kinds. Note that host offload kinds
  /// are only set if the action is a dependence to a host offload action.
  bool isHostOffloading(OffloadKind OKind) const {
    return ActiveOffloadKindMask & OKind;
  }
  bool isDeviceOffloading(OffloadKind OKind) const {
    return OffloadingDeviceKind == OKind;
  }
  bool isOffloading(OffloadKind OKind) const {
    return isHostOffloading(OKind) || isDeviceOffloading(OKind);
  }
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
  BindArchAction(Action *Input, const char *ArchName);

  const char *getArchName() const { return ArchName; }

  static bool classof(const Action *A) {
    return A->getKind() == BindArchClass;
  }
};

/// An offload action combines host or/and device actions according to the
/// programming model implementation needs and propagates the offloading kind to
/// its dependences.
class OffloadAction final : public Action {
  virtual void anchor();

public:
  /// Type used to communicate device actions. It associates bound architecture,
  /// toolchain, and offload kind to each action.
  class DeviceDependences final {
  public:
    typedef SmallVector<const ToolChain *, 3> ToolChainList;
    typedef SmallVector<const char *, 3> BoundArchList;
    typedef SmallVector<OffloadKind, 3> OffloadKindList;

  private:
    // Lists that keep the information for each dependency. All the lists are
    // meant to be updated in sync. We are adopting separate lists instead of a
    // list of structs, because that simplifies forwarding the actions list to
    // initialize the inputs of the base Action class.

    /// The dependence actions.
    ActionList DeviceActions;
    /// The offloading toolchains that should be used with the action.
    ToolChainList DeviceToolChains;
    /// The architectures that should be used with this action.
    BoundArchList DeviceBoundArchs;
    /// The offload kind of each dependence.
    OffloadKindList DeviceOffloadKinds;

  public:
    /// Add a action along with the associated toolchain, bound arch, and
    /// offload kind.
    void add(Action &A, const ToolChain &TC, const char *BoundArch,
             OffloadKind OKind);

    /// Get each of the individual arrays.
    const ActionList &getActions() const { return DeviceActions; };
    const ToolChainList &getToolChains() const { return DeviceToolChains; };
    const BoundArchList &getBoundArchs() const { return DeviceBoundArchs; };
    const OffloadKindList &getOffloadKinds() const {
      return DeviceOffloadKinds;
    };
  };

  /// Type used to communicate host actions. It associates bound architecture,
  /// toolchain, and offload kinds to the host action.
  class HostDependence final {
    /// The dependence action.
    Action &HostAction;
    /// The offloading toolchain that should be used with the action.
    const ToolChain &HostToolChain;
    /// The architectures that should be used with this action.
    const char *HostBoundArch = nullptr;
    /// The offload kind of each dependence.
    unsigned HostOffloadKinds = 0u;

  public:
    HostDependence(Action &A, const ToolChain &TC, const char *BoundArch,
                   const unsigned OffloadKinds)
        : HostAction(A), HostToolChain(TC), HostBoundArch(BoundArch),
          HostOffloadKinds(OffloadKinds){};
    /// Constructor version that obtains the offload kinds from the device
    /// dependencies.
    HostDependence(Action &A, const ToolChain &TC, const char *BoundArch,
                   const DeviceDependences &DDeps);
    Action *getAction() const { return &HostAction; };
    const ToolChain *getToolChain() const { return &HostToolChain; };
    const char *getBoundArch() const { return HostBoundArch; };
    unsigned getOffloadKinds() const { return HostOffloadKinds; };
  };

  typedef llvm::function_ref<void(Action *, const ToolChain *, const char *)>
      OffloadActionWorkTy;

private:
  /// The host offloading toolchain that should be used with the action.
  const ToolChain *HostTC = nullptr;

  /// The tool chains associated with the list of actions.
  DeviceDependences::ToolChainList DevToolChains;

public:
  OffloadAction(const HostDependence &HDep);
  OffloadAction(const DeviceDependences &DDeps, types::ID Ty);
  OffloadAction(const HostDependence &HDep, const DeviceDependences &DDeps);

  /// Execute the work specified in \a Work on the host dependence.
  void doOnHostDependence(const OffloadActionWorkTy &Work) const;

  /// Execute the work specified in \a Work on each device dependence.
  void doOnEachDeviceDependence(const OffloadActionWorkTy &Work) const;

  /// Execute the work specified in \a Work on each dependence.
  void doOnEachDependence(const OffloadActionWorkTy &Work) const;

  /// Execute the work specified in \a Work on each host or device dependence if
  /// \a IsHostDependenceto is true or false, respectively.
  void doOnEachDependence(bool IsHostDependence,
                          const OffloadActionWorkTy &Work) const;

  /// Return true if the action has a host dependence.
  bool hasHostDependence() const;

  /// Return the host dependence of this action. This function is only expected
  /// to be called if the host dependence exists.
  Action *getHostDependence() const;

  /// Return true if the action has a single device dependence. If \a
  /// DoNotConsiderHostActions is set, ignore the host dependence, if any, while
  /// accounting for the number of dependences.
  bool hasSingleDeviceDependence(bool DoNotConsiderHostActions = false) const;

  /// Return the single device dependence of this action. This function is only
  /// expected to be called if a single device dependence exists. If \a
  /// DoNotConsiderHostActions is set, a host dependence is allowed.
  Action *
  getSingleDeviceDependence(bool DoNotConsiderHostActions = false) const;

  static bool classof(const Action *A) { return A->getKind() == OffloadClass; }
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
  void anchor() override;
public:
  PreprocessJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == PreprocessJobClass;
  }
};

class PrecompileJobAction : public JobAction {
  void anchor() override;
public:
  PrecompileJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == PrecompileJobClass;
  }
};

class AnalyzeJobAction : public JobAction {
  void anchor() override;
public:
  AnalyzeJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == AnalyzeJobClass;
  }
};

class MigrateJobAction : public JobAction {
  void anchor() override;
public:
  MigrateJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == MigrateJobClass;
  }
};

class CompileJobAction : public JobAction {
  void anchor() override;
public:
  CompileJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == CompileJobClass;
  }
};

class BackendJobAction : public JobAction {
  void anchor() override;
public:
  BackendJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == BackendJobClass;
  }
};

class AssembleJobAction : public JobAction {
  void anchor() override;
public:
  AssembleJobAction(Action *Input, types::ID OutputType);

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
  VerifyJobAction(ActionClass Kind, Action *Input, types::ID Type);
  static bool classof(const Action *A) {
    return A->getKind() == VerifyDebugInfoJobClass ||
           A->getKind() == VerifyPCHJobClass;
  }
};

class VerifyDebugInfoJobAction : public VerifyJobAction {
  void anchor() override;
public:
  VerifyDebugInfoJobAction(Action *Input, types::ID Type);
  static bool classof(const Action *A) {
    return A->getKind() == VerifyDebugInfoJobClass;
  }
};

class VerifyPCHJobAction : public VerifyJobAction {
  void anchor() override;
public:
  VerifyPCHJobAction(Action *Input, types::ID Type);
  static bool classof(const Action *A) {
    return A->getKind() == VerifyPCHJobClass;
  }
};

} // end namespace driver
} // end namespace clang

#endif
