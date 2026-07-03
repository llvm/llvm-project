//===--- OpenACCKinds.h - OpenACC Enums -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines some OpenACC-specific enums and functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OPENACCKINDS_H
#define LLVM_CLANG_BASIC_OPENACCKINDS_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Represents the Construct/Directive kind of a pragma directive. Note the
// OpenACC standard is inconsistent between calling these Construct vs
// Directive, but we're calling it a Directive to be consistent with OpenMP.
enum class OpenACCDirectiveKind : uint8_t {
  // Compute Constructs.
  Parallel,
  Serial,
  Kernels,

  // Data Environment. "enter data" and "exit data" are also referred to in the
  // Executable Directives section, but just as a back reference to the Data
  // Environment.
  Data,
  EnterData,
  ExitData,
  HostData,

  // Misc.
  Loop,
  Cache,

  // Combined Constructs.
  ParallelLoop,
  SerialLoop,
  KernelsLoop,

  // Atomic Construct.
  Atomic,

  // Declare Directive.
  Declare,

  // Executable Directives. "wait" is first referred to here, but ends up being
  // in its own section after "routine".
  Init,
  Shutdown,
  Set,
  Update,
  Wait,

  // Procedure Calls in Compute Regions.
  Routine,

  // Invalid.
  Invalid,
};

template <typename StreamTy>
inline StreamTy &printOpenACCDirectiveKind(StreamTy &Out,
                                           OpenACCDirectiveKind K) {
  switch (K) {
  case OpenACCDirectiveKind::Parallel:
    return Out << "parallel";

  case OpenACCDirectiveKind::Serial:
    return Out << "serial";

  case OpenACCDirectiveKind::Kernels:
    return Out << "kernels";

  case OpenACCDirectiveKind::Data:
    return Out << "data";

  case OpenACCDirectiveKind::EnterData:
    return Out << "enter data";

  case OpenACCDirectiveKind::ExitData:
    return Out << "exit data";

  case OpenACCDirectiveKind::HostData:
    return Out << "host_data";

  case OpenACCDirectiveKind::Loop:
    return Out << "loop";

  case OpenACCDirectiveKind::Cache:
    return Out << "cache";

  case OpenACCDirectiveKind::ParallelLoop:
    return Out << "parallel loop";

  case OpenACCDirectiveKind::SerialLoop:
    return Out << "serial loop";

  case OpenACCDirectiveKind::KernelsLoop:
    return Out << "kernels loop";

  case OpenACCDirectiveKind::Atomic:
    return Out << "atomic";

  case OpenACCDirectiveKind::Declare:
    return Out << "declare";

  case OpenACCDirectiveKind::Init:
    return Out << "init";

  case OpenACCDirectiveKind::Shutdown:
    return Out << "shutdown";

  case OpenACCDirectiveKind::Set:
    return Out << "set";

  case OpenACCDirectiveKind::Update:
    return Out << "update";

  case OpenACCDirectiveKind::Wait:
    return Out << "wait";

  case OpenACCDirectiveKind::Routine:
    return Out << "routine";

  case OpenACCDirectiveKind::Invalid:
    return Out << "<invalid>";
  }
  llvm_unreachable("Uncovered directive kind");
}

inline const StreamingDiagnostic &operator<<(const StreamingDiagnostic &Out,
                                             OpenACCDirectiveKind K) {
  return printOpenACCDirectiveKind(Out, K);
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                                     OpenACCDirectiveKind K) {
  return printOpenACCDirectiveKind(Out, K);
}

inline bool isOpenACCComputeDirectiveKind(OpenACCDirectiveKind K) {
  return K == OpenACCDirectiveKind::Parallel ||
         K == OpenACCDirectiveKind::Serial ||
         K == OpenACCDirectiveKind::Kernels;
}

inline bool isOpenACCCombinedDirectiveKind(OpenACCDirectiveKind K) {
  return K == OpenACCDirectiveKind::ParallelLoop ||
         K == OpenACCDirectiveKind::SerialLoop ||
         K == OpenACCDirectiveKind::KernelsLoop;
}

// Tests 'K' to see if it is 'data', 'host_data', 'enter data', or 'exit data'.
inline bool isOpenACCDataDirectiveKind(OpenACCDirectiveKind K) {
  return K == OpenACCDirectiveKind::Data ||
         K == OpenACCDirectiveKind::EnterData ||
         K == OpenACCDirectiveKind::ExitData ||
         K == OpenACCDirectiveKind::HostData;
}

enum class OpenACCAtomicKind : uint8_t {
  Read,
  Write,
  Update,
  Capture,
  None,
};

template <typename StreamTy>
inline StreamTy &printOpenACCAtomicKind(StreamTy &Out, OpenACCAtomicKind AK) {
  switch (AK) {
  case OpenACCAtomicKind::Read:
    return Out << "read";
  case OpenACCAtomicKind::Write:
    return Out << "write";
  case OpenACCAtomicKind::Update:
    return Out << "update";
  case OpenACCAtomicKind::Capture:
    return Out << "capture";
  case OpenACCAtomicKind::None:
    return Out << "<none>";
  }
  llvm_unreachable("unknown atomic kind");
}
inline const StreamingDiagnostic &operator<<(const StreamingDiagnostic &Out,
                                             OpenACCAtomicKind AK) {
  return printOpenACCAtomicKind(Out, AK);
}
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                                     OpenACCAtomicKind AK) {
  return printOpenACCAtomicKind(Out, AK);
}

/// Represents the kind of an OpenACC clause. Sorted alphabetically, since this
/// order ends up influencing the sorting of the list diagnostic.
enum class OpenACCClauseKind : uint8_t {
  /// 'async' clause, allowed on Compute, Data, 'update', 'wait', and Combined
  /// constructs.
  Async,
  /// 'attach' clause, allowed on Compute and Combined constructs, plus 'data'
  /// and 'enter data'.
  Attach,
  /// 'auto' clause, allowed on 'loop' directives.
  Auto,
  /// 'bind' clause, allowed on routine constructs.
  Bind,
  /// 'collapse' clause, allowed on 'loop' and Combined constructs.
  Collapse,
  /// 'copy' clause, allowed on Compute and Combined Constructs, plus 'data' and
  /// 'declare'.
  Copy,
  /// 'copy' clause alias 'pcopy'.  Preserved for diagnostic purposes.
  PCopy,
  /// 'copy' clause alias 'present_or_copy'.  Preserved for diagnostic purposes.
  PresentOrCopy,
  /// 'copyin' clause, allowed on Compute and Combined constructs, plus 'data',
  /// 'enter data', and 'declare'.
  CopyIn,
  /// 'copyin' clause alias 'pcopyin'.  Preserved for diagnostic purposes.
  PCopyIn,
  /// 'copyin' clause alias 'present_or_copyin'.  Preserved for diagnostic
  /// purposes.
  PresentOrCopyIn,
  /// 'copyout' clause, allowed on Compute and Combined constructs, plus 'data',
  /// 'exit data', and 'declare'.
  CopyOut,
  /// 'copyout' clause alias 'pcopyout'.  Preserved for diagnostic purposes.
  PCopyOut,
  /// 'copyout' clause alias 'present_or_copyout'.  Preserved for diagnostic
  /// purposes.
  PresentOrCopyOut,
  /// 'create' clause, allowed on Compute and Combined constructs, plus 'data',
  /// 'enter data', and 'declare'.
  Create,
  /// 'create' clause alias 'pcreate'.  Preserved for diagnostic purposes.
  PCreate,
  /// 'create' clause alias 'present_or_create'.  Preserved for diagnostic
  /// purposes.
  PresentOrCreate,
  /// 'default' clause, allowed on parallel, serial, kernel (and compound)
  /// constructs.
  Default,
  /// 'default_async' clause, allowed on 'set' construct.
  DefaultAsync,
  /// 'delete' clause, allowed on the 'exit data' construct.
  Delete,
  /// 'detach' clause, allowed on the 'exit data' construct.
  Detach,
  /// 'device' clause, allowed on the 'update' construct.
  Device,
  /// 'device_num' clause, allowed on 'init', 'shutdown', and 'set' constructs.
  DeviceNum,
  /// 'deviceptr' clause, allowed on Compute and Combined Constructs, plus
  /// 'data' and 'declare'.
  DevicePtr,
  /// 'device_resident' clause, allowed on the 'declare' construct.
  DeviceResident,
  /// 'device_type' clause, allowed on Compute, 'data', 'init', 'shutdown',
  /// 'set', update', 'loop', 'routine', and Combined constructs.
  DeviceType,
  /// 'dtype' clause, an alias for 'device_type', stored separately for
  /// diagnostic purposes.
  DType,
  /// 'finalize' clause, allowed on 'exit data' directive.
  Finalize,
  /// 'firstprivate' clause, allowed on 'parallel', 'serial', 'parallel loop',
  /// and 'serial loop' constructs.
  FirstPrivate,
  /// 'gang' clause, allowed on 'loop' and Combined constructs.
  Gang,
  /// 'host' clause, allowed on 'update' construct.
  Host,
  /// 'if' clause, allowed on all the Compute Constructs, Data Constructs,
  /// Executable Constructs, and Combined Constructs.
  If,
  /// 'if_present' clause, allowed on 'host_data' and 'update' directives.
  IfPresent,
  /// 'independent' clause, allowed on 'loop' directives.
  Independent,
  /// 'link' clause, allowed on 'declare' construct.
  Link,
  /// 'no_create' clause, allowed on allowed on Compute and Combined constructs,
  /// plus 'data'.
  NoCreate,
  /// 'nohost' clause, allowed on 'routine' directives.
  NoHost,
  /// 'num_gangs' clause, allowed on 'parallel', 'kernels', parallel loop', and
  /// 'kernels loop' constructs.
  NumGangs,
  /// 'num_workers' clause, allowed on 'parallel', 'kernels', parallel loop',
  /// and 'kernels loop' constructs.
  NumWorkers,
  /// 'present' clause, allowed on Compute and Combined constructs, plus 'data'
  /// and 'declare'.
  Present,
  /// 'private' clause, allowed on 'parallel', 'serial', 'loop', 'parallel
  /// loop', and 'serial loop' constructs.
  Private,
  /// 'reduction' clause, allowed on Parallel, Serial, Loop, and the combined
  /// constructs.
  Reduction,
  /// 'self' clause, allowed on Compute and Combined Constructs, plus 'update'.
  Self,
  /// 'seq' clause, allowed on 'loop' and 'routine' directives.
  Seq,
  /// 'tile' clause, allowed on 'loop' and Combined constructs.
  Tile,
  /// 'use_device' clause, allowed on 'host_data' construct.
  UseDevice,
  /// 'vector' clause, allowed on 'loop', Combined, and 'routine' directives.
  Vector,
  /// 'vector_length' clause, allowed on 'parallel', 'kernels', 'parallel loop',
  /// and 'kernels loop' constructs.
  VectorLength,
  /// 'wait' clause, allowed on Compute, Data, 'update', and Combined
  /// constructs.
  Wait,
  /// 'worker' clause, allowed on 'loop', Combined, and 'routine' directives.
  Worker,

  /// 'shortloop' is represented in the ACC.td file, but isn't present in the
  /// standard. This appears to be an old extension for the nvidia fortran
  // compiler, but seemingly not elsewhere. Put it here as a placeholder, but it
  // is never expected to be generated.
  Shortloop,
  /// Represents an invalid clause, for the purposes of parsing. Should be
  /// 'last'.
  Invalid,
};

template <typename StreamTy>
inline StreamTy &printOpenACCClauseKind(StreamTy &Out, OpenACCClauseKind K) {
  switch (K) {
  case OpenACCClauseKind::Finalize:
    return Out << "finalize";

  case OpenACCClauseKind::IfPresent:
    return Out << "if_present";

  case OpenACCClauseKind::Seq:
    return Out << "seq";

  case OpenACCClauseKind::Independent:
    return Out << "independent";

  case OpenACCClauseKind::Auto:
    return Out << "auto";

  case OpenACCClauseKind::Worker:
    return Out << "worker";

  case OpenACCClauseKind::Vector:
    return Out << "vector";

  case OpenACCClauseKind::NoHost:
    return Out << "nohost";

  case OpenACCClauseKind::Default:
    return Out << "default";

  case OpenACCClauseKind::If:
    return Out << "if";

  case OpenACCClauseKind::Self:
    return Out << "self";

  case OpenACCClauseKind::Copy:
    return Out << "copy";

  case OpenACCClauseKind::PCopy:
    return Out << "pcopy";

  case OpenACCClauseKind::PresentOrCopy:
    return Out << "present_or_copy";

  case OpenACCClauseKind::UseDevice:
    return Out << "use_device";

  case OpenACCClauseKind::Attach:
    return Out << "attach";

  case OpenACCClauseKind::Delete:
    return Out << "delete";

  case OpenACCClauseKind::Detach:
    return Out << "detach";

  case OpenACCClauseKind::Device:
    return Out << "device";

  case OpenACCClauseKind::DevicePtr:
    return Out << "deviceptr";

  case OpenACCClauseKind::DeviceResident:
    return Out << "device_resident";

  case OpenACCClauseKind::FirstPrivate:
    return Out << "firstprivate";

  case OpenACCClauseKind::Host:
    return Out << "host";

  case OpenACCClauseKind::Link:
    return Out << "link";

  case OpenACCClauseKind::NoCreate:
    return Out << "no_create";

  case OpenACCClauseKind::Present:
    return Out << "present";

  case OpenACCClauseKind::Private:
    return Out << "private";

  case OpenACCClauseKind::CopyOut:
    return Out << "copyout";

  case OpenACCClauseKind::PCopyOut:
    return Out << "pcopyout";

  case OpenACCClauseKind::PresentOrCopyOut:
    return Out << "present_or_copyout";

  case OpenACCClauseKind::CopyIn:
    return Out << "copyin";

  case OpenACCClauseKind::PCopyIn:
    return Out << "pcopyin";

  case OpenACCClauseKind::PresentOrCopyIn:
    return Out << "present_or_copyin";

  case OpenACCClauseKind::Create:
    return Out << "create";

  case OpenACCClauseKind::PCreate:
    return Out << "pcreate";

  case OpenACCClauseKind::PresentOrCreate:
    return Out << "present_or_create";

  case OpenACCClauseKind::Reduction:
    return Out << "reduction";

  case OpenACCClauseKind::Collapse:
    return Out << "collapse";

  case OpenACCClauseKind::Bind:
    return Out << "bind";

  case OpenACCClauseKind::VectorLength:
    return Out << "vector_length";

  case OpenACCClauseKind::NumGangs:
    return Out << "num_gangs";

  case OpenACCClauseKind::NumWorkers:
    return Out << "num_workers";

  case OpenACCClauseKind::DeviceNum:
    return Out << "device_num";

  case OpenACCClauseKind::DefaultAsync:
    return Out << "default_async";

  case OpenACCClauseKind::DeviceType:
    return Out << "device_type";

  case OpenACCClauseKind::DType:
    return Out << "dtype";

  case OpenACCClauseKind::Async:
    return Out << "async";

  case OpenACCClauseKind::Tile:
    return Out << "tile";

  case OpenACCClauseKind::Gang:
    return Out << "gang";

  case OpenACCClauseKind::Wait:
    return Out << "wait";

  case OpenACCClauseKind::Shortloop:
    llvm_unreachable("Shortloop shouldn't be generated in clang");
    [[fallthrough]];
  case OpenACCClauseKind::Invalid:
    return Out << "<invalid>";
  }
  llvm_unreachable("Uncovered clause kind");
}

inline const StreamingDiagnostic &operator<<(const StreamingDiagnostic &Out,
                                             OpenACCClauseKind K) {
  return printOpenACCClauseKind(Out, K);
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                                     OpenACCClauseKind K) {
  return printOpenACCClauseKind(Out, K);
}

enum class OpenACCDefaultClauseKind : uint8_t {
  /// 'none' option.
  None,
  /// 'present' option.
  Present,
  /// Not a valid option.
  Invalid,
};

template <typename StreamTy>
inline StreamTy &printOpenACCDefaultClauseKind(StreamTy &Out,
                                               OpenACCDefaultClauseKind K) {
  switch (K) {
  case OpenACCDefaultClauseKind::None:
    return Out << "none";
  case OpenACCDefaultClauseKind::Present:
    return Out << "present";
  case OpenACCDefaultClauseKind::Invalid:
    return Out << "<invalid>";
  }
  llvm_unreachable("Unknown OpenACCDefaultClauseKind enum");
}

inline const StreamingDiagnostic &operator<<(const StreamingDiagnostic &Out,
                                             OpenACCDefaultClauseKind K) {
  return printOpenACCDefaultClauseKind(Out, K);
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                                     OpenACCDefaultClauseKind K) {
  return printOpenACCDefaultClauseKind(Out, K);
}

enum class OpenACCReductionOperator : uint8_t {
  /// '+'.
  Addition,
  /// '*'.
  Multiplication,
  /// 'max'.
  Max,
  /// 'min'.
  Min,
  /// '&'.
  BitwiseAnd,
  /// '|'.
  BitwiseOr,
  /// '^'.
  BitwiseXOr,
  /// '&&'.
  And,
  /// '||'.
  Or,
  /// Invalid Reduction Clause Kind.
  Invalid,
};

template <typename StreamTy>
inline StreamTy &printOpenACCReductionOperator(StreamTy &Out,
                                               OpenACCReductionOperator Op) {
  switch (Op) {
  case OpenACCReductionOperator::Addition:
    return Out << "+";
  case OpenACCReductionOperator::Multiplication:
    return Out << "*";
  case OpenACCReductionOperator::Max:
    return Out << "max";
  case OpenACCReductionOperator::Min:
    return Out << "min";
  case OpenACCReductionOperator::BitwiseAnd:
    return Out << "&";
  case OpenACCReductionOperator::BitwiseOr:
    return Out << "|";
  case OpenACCReductionOperator::BitwiseXOr:
    return Out << "^";
  case OpenACCReductionOperator::And:
    return Out << "&&";
  case OpenACCReductionOperator::Or:
    return Out << "||";
  case OpenACCReductionOperator::Invalid:
    return Out << "<invalid>";
  }
  llvm_unreachable("Unknown reduction operator kind");
}
inline const StreamingDiagnostic &operator<<(const StreamingDiagnostic &Out,
                                             OpenACCReductionOperator Op) {
  return printOpenACCReductionOperator(Out, Op);
}
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                                     OpenACCReductionOperator Op) {
  return printOpenACCReductionOperator(Out, Op);
}

enum class OpenACCGangKind : uint8_t {
  /// num:
  Num,
  /// dim:
  Dim,
  /// static:
  Static
};

template <typename StreamTy>
inline StreamTy &printOpenACCGangKind(StreamTy &Out, OpenACCGangKind GK) {
  switch (GK) {
  case OpenACCGangKind::Num:
    return Out << "num";
  case OpenACCGangKind::Dim:
    return Out << "dim";
  case OpenACCGangKind::Static:
    return Out << "static";
  }
  llvm_unreachable("unknown gang kind");
}
inline const StreamingDiagnostic &operator<<(const StreamingDiagnostic &Out,
                                             OpenACCGangKind Op) {
  return printOpenACCGangKind(Out, Op);
}
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                                     OpenACCGangKind Op) {
  return printOpenACCGangKind(Out, Op);
}

// Represents the 'modifier' of a 'modifier-list', as applied to copy, copyin,
// copyout, and create. Implemented as a 'bitmask'.
// Note: This attempts to synchronize with mlir::acc::DataClauseModifier,
// however has to store `Always` separately(whereas MLIR has it as AlwaysIn &
// AlwaysOut). However, we keep them in sync so that we can cast between them.
enum class OpenACCModifierKind : uint8_t {
  Invalid = 0,
  Zero = 1 << 0,
  Readonly = 1 << 1,
  AlwaysIn = 1 << 2,
  AlwaysOut = 1 << 3,
  Capture = 1 << 4,
  Always = 1 << 5,
  LLVM_MARK_AS_BITMASK_ENUM(Always)
};

inline bool isOpenACCModifierBitSet(OpenACCModifierKind List,
                                    OpenACCModifierKind Bit) {
  return (List & Bit) != OpenACCModifierKind::Invalid;
}

template <typename StreamTy>
inline StreamTy &printOpenACCModifierKind(StreamTy &Out,
                                          OpenACCModifierKind Mods) {
  if (Mods == OpenACCModifierKind::Invalid)
    return Out << "<invalid>";

  bool First = true;

  if (isOpenACCModifierBitSet(Mods, OpenACCModifierKind::Always)) {
    Out << "always";
    First = false;
  }

  if (isOpenACCModifierBitSet(Mods, OpenACCModifierKind::AlwaysIn)) {
    if (!First)
      Out << ", ";
    Out << "alwaysin";
    First = false;
  }

  if (isOpenACCModifierBitSet(Mods, OpenACCModifierKind::AlwaysOut)) {
    if (!First)
      Out << ", ";
    Out << "alwaysout";
    First = false;
  }

  if (isOpenACCModifierBitSet(Mods, OpenACCModifierKind::Readonly)) {
    if (!First)
      Out << ", ";
    Out << "readonly";
    First = false;
  }

  if (isOpenACCModifierBitSet(Mods, OpenACCModifierKind::Zero)) {
    if (!First)
      Out << ", ";
    Out << "zero";
    First = false;
  }

  if (isOpenACCModifierBitSet(Mods, OpenACCModifierKind::Capture)) {
    if (!First)
      Out << ", ";
    Out << "capture";
    First = false;
  }
  return Out;
}
inline const StreamingDiagnostic &operator<<(const StreamingDiagnostic &Out,
                                             OpenACCModifierKind Op) {
  return printOpenACCModifierKind(Out, Op);
}
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                                     OpenACCModifierKind Op) {
  return printOpenACCModifierKind(Out, Op);
}
} // namespace clang

#endif // LLVM_CLANG_BASIC_OPENACCKINDS_H
