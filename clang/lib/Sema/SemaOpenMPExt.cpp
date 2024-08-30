#include "SemaOpenMPExt.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprOpenMP.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Frontend/OpenMP/ClauseT.h"
#include "llvm/Frontend/OpenMP/ConstructDecompositionT.h"

#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

using namespace clang;

#define MAKE_EMPTY_CLASS(cls, from_cls)                                        \
  cls make(const from_cls &) {                                                 \
    static_assert(cls::EmptyTrait::value);                                     \
    return cls{};                                                              \
  }                                                                            \
  [[maybe_unused]] extern int xyzzy_semicolon_absorber

#define MS(x, y) CLAUSET_SCOPED_ENUM_MEMBER_CONVERT(x, y)
#define MU(x, y) CLAUSET_UNSCOPED_ENUM_MEMBER_CONVERT(x, y)

namespace ext {
std::optional<clause::Default::DataSharingAttribute>
conv(llvm::omp::DefaultKind T) {
  // OMP_DEFAULT_unknown -> std::nullopt
  using FromTy = llvm::omp::DefaultKind;
  using ToTy = clause::Default::DataSharingAttribute;

  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::OMP_DEFAULT_firstprivate, ToTy::Firstprivate},
      {FromTy::OMP_DEFAULT_none, ToTy::None},
      {FromTy::OMP_DEFAULT_private, ToTy::Private},
      {FromTy::OMP_DEFAULT_shared, ToTy::Shared},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::ProcBind::AffinityPolicy>
conv(llvm::omp::ProcBindKind T) {
  // OMP_PROC_BIND_default -> assert
  // OMP_PROC_BIND_unknown -> std::nullopt
  using FromTy = llvm::omp::ProcBindKind;
  using ToTy = clause::ProcBind::AffinityPolicy;

  assert(T != FromTy::OMP_PROC_BIND_default && "Unexpected kind");
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::OMP_PROC_BIND_close, ToTy::Close},
      {FromTy::OMP_PROC_BIND_master, ToTy::Master},
      {FromTy::OMP_PROC_BIND_primary, ToTy::Primary},
      {FromTy::OMP_PROC_BIND_spread, ToTy::Spread},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::At::ActionTime> conv(OpenMPAtClauseKind T) {
  // OMPC_AT_unknown -> std::nullopt
  using FromTy = OpenMPAtClauseKind;
  using ToTy = clause::At::ActionTime;

  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_AT_compilation, ToTy::Compilation},
      {OMPC_AT_execution, ToTy::Execution},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<MemoryOrder> conv(OpenMPAtomicDefaultMemOrderClauseKind T) {
  // OMPC_ATOMIC_DEFAULT_MEM_ORDER_unknown -> std::nullopt
  using FromTy = OpenMPAtomicDefaultMemOrderClauseKind;
  using ToTy = MemoryOrder;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_ATOMIC_DEFAULT_MEM_ORDER_acq_rel, ToTy::AcqRel},
      {OMPC_ATOMIC_DEFAULT_MEM_ORDER_relaxed, ToTy::Relaxed},
      {OMPC_ATOMIC_DEFAULT_MEM_ORDER_seq_cst, ToTy::SeqCst},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Bind::Binding> conv(OpenMPBindClauseKind T) {
  // OMPC_BIND_unknown -> std::nullopt
  using FromTy = OpenMPBindClauseKind;
  using ToTy = clause::Bind::Binding;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_BIND_parallel, ToTy::Parallel},
      {OMPC_BIND_teams, ToTy::Teams},
      {OMPC_BIND_thread, ToTy::Thread},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Defaultmap::VariableCategory>
conv(OpenMPDefaultmapClauseKind T) {
  // OMPC_DEFAULTMAP_all -> assert
  // OMPC_DEFAULTMAP_unknown -> std::nullopt
  using FromTy = OpenMPDefaultmapClauseKind;
  using ToTy = clause::Defaultmap::VariableCategory;
  assert(T != OMPC_DEFAULTMAP_all && "Unexpected kind");
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_DEFAULTMAP_aggregate, ToTy::Aggregate},
      {OMPC_DEFAULTMAP_pointer, ToTy::Pointer},
      {OMPC_DEFAULTMAP_scalar, ToTy::Scalar},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Defaultmap::ImplicitBehavior>
conv(OpenMPDefaultmapClauseModifier T) {
  // OMPC_DEFAULTMAP_MODIFIER_unknown -> std::nullopt
  // OMPC_DEFAULTMAP_MODIFIER_last -> std::nullopt
  using FromTy = OpenMPDefaultmapClauseModifier;
  using ToTy = clause::Defaultmap::ImplicitBehavior;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_DEFAULTMAP_MODIFIER_alloc, ToTy::Alloc},
      {OMPC_DEFAULTMAP_MODIFIER_default, ToTy::Default},
      {OMPC_DEFAULTMAP_MODIFIER_firstprivate, ToTy::Firstprivate},
      {OMPC_DEFAULTMAP_MODIFIER_from, ToTy::From},
      {OMPC_DEFAULTMAP_MODIFIER_none, ToTy::None},
      {OMPC_DEFAULTMAP_MODIFIER_present, ToTy::Present},
      {OMPC_DEFAULTMAP_MODIFIER_to, ToTy::To},
      {OMPC_DEFAULTMAP_MODIFIER_tofrom, ToTy::Tofrom},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Device::DeviceModifier>
conv(OpenMPDeviceClauseModifier T) {
  // OMPC_DEVICE_unknown -> std::nullopt
  using FromTy = OpenMPDeviceClauseModifier;
  using ToTy = clause::Device::DeviceModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_DEVICE_ancestor, ToTy::Ancestor},
      {OMPC_DEVICE_device_num, ToTy::DeviceNum},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::DistSchedule::Kind> conv(OpenMPDistScheduleClauseKind T) {
  // OMPC_DIST_SCHEDULE_unknown -> std::nullopt
  using FromTy = OpenMPDistScheduleClauseKind;
  using ToTy = clause::DistSchedule::Kind;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_DIST_SCHEDULE_static, ToTy::Static},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Prescriptiveness> conv(OpenMPGrainsizeClauseModifier T) {
  // OMPC_GRAINSIZE_unknown -> std::nullopt
  using FromTy = OpenMPGrainsizeClauseModifier;
  using ToTy = clause::Prescriptiveness;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_GRAINSIZE_strict, ToTy::Strict},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Lastprivate::LastprivateModifier>
conv(OpenMPLastprivateModifier T) {
  // OMPC_LASTPRIVATE_unknown -> std::nullopt
  using FromTy = OpenMPLastprivateModifier;
  using ToTy = clause::Lastprivate::LastprivateModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_LASTPRIVATE_conditional, ToTy::Conditional},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Linear::LinearModifier> conv(OpenMPLinearClauseKind T) {
  // OMPC_LINEAR_step -> assert
  // OMPC_LINEAR_unknown -> std::nullopt
  using FromTy = OpenMPLinearClauseKind;
  using ToTy = clause::Linear::LinearModifier;
  assert(T != OMPC_LINEAR_step && "Unexpected kind");
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_LINEAR_ref, ToTy::Ref},
      {OMPC_LINEAR_uval, ToTy::Uval},
      {OMPC_LINEAR_val, ToTy::Val},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Map::MapType> conv(OpenMPMapClauseKind T) {
  // OMPC_MAP_unknown -> std::nullopt
  using FromTy = OpenMPMapClauseKind;
  using ToTy = clause::Map::MapType;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_MAP_alloc, ToTy::Alloc},   {OMPC_MAP_to, ToTy::To},
      {OMPC_MAP_from, ToTy::From},     {OMPC_MAP_tofrom, ToTy::Tofrom},
      {OMPC_MAP_delete, ToTy::Delete}, {OMPC_MAP_release, ToTy::Release},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Map::MapTypeModifier> conv(OpenMPMapModifierKind T) {
  // OMPC_MAP_MODIFIER_last -> std::nullopt
  // OMPC_MAP_MODIFIER_unknown -> std::nullopt
  using FromTy = OpenMPMapModifierKind;
  using ToTy = clause::Map::MapTypeModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_MAP_MODIFIER_always, ToTy::Always},
      {OMPC_MAP_MODIFIER_close, ToTy::Close},
      {OMPC_MAP_MODIFIER_present, ToTy::Present},
      {OMPC_MAP_MODIFIER_ompx_hold, ToTy::OmpxHold},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Prescriptiveness> conv(OpenMPNumTasksClauseModifier T) {
  // OMPC_NUMTASKS_unknown -> std::nullopt
  using FromTy = OpenMPNumTasksClauseModifier;
  using ToTy = clause::Prescriptiveness;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_NUMTASKS_strict, ToTy::Strict},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Order::Ordering> conv(OpenMPOrderClauseKind T) {
  // OMPC_ORDER_unknown -> std::nullopt
  using FromTy = OpenMPOrderClauseKind;
  using ToTy = clause::Order::Ordering;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_ORDER_concurrent, ToTy::Concurrent},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Order::OrderModifier> conv(OpenMPOrderClauseModifier T) {
  // OMPC_ORDER_MODIFIER_last -> std::nullopt
  // OMPC_ORDER_MODIFIER_unknown -> std::nullopt
  using FromTy = OpenMPOrderClauseModifier;
  using ToTy = clause::Order::OrderModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_ORDER_MODIFIER_reproducible, ToTy::Reproducible},
      {OMPC_ORDER_MODIFIER_unconstrained, ToTy::Unconstrained},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Reduction::ReductionModifier>
conv(OpenMPReductionClauseModifier T) {
  // OMPC_REDUCTION_unknown -> std::nullopt
  using FromTy = OpenMPReductionClauseModifier;
  using ToTy = clause::Reduction::ReductionModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_REDUCTION_default, ToTy::Default},
      {OMPC_REDUCTION_inscan, ToTy::Inscan},
      {OMPC_REDUCTION_task, ToTy::Task},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

std::optional<clause::Severity::SevLevel> conv(OpenMPSeverityClauseKind T) {
  // OMPC_SEVERITY_unknown -> std::nullopt
  using FromTy = OpenMPSeverityClauseKind;
  using ToTy = clause::Severity::SevLevel;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OMPC_SEVERITY_fatal, ToTy::Fatal},
      {OMPC_SEVERITY_warning, ToTy::Warning},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  return std::nullopt;
}

clause::DefinedOperator::IntrinsicOperator conv(OverloadedOperatorKind T) {
  using FromTy = OverloadedOperatorKind;
  using ToTy = clause::DefinedOperator::IntrinsicOperator;

  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {OO_Plus, ToTy::Add},      {OO_Minus, ToTy::Subtract},
      {OO_Star, ToTy::Multiply}, {OO_Amp, ToTy::AND},
      {OO_Pipe, ToTy::OR},       {OO_Caret, ToTy::NEQV},
      {OO_AmpAmp, ToTy::AND},    {OO_PipePipe, ToTy::OR},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

// Reverse conversions

llvm::omp::DefaultKind vnoc(clause::Default::DataSharingAttribute T) {
  using FromTy = clause::Default::DataSharingAttribute;
  using ToTy = llvm::omp::DefaultKind;

  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Firstprivate, ToTy::OMP_DEFAULT_firstprivate},
      {FromTy::None, ToTy::OMP_DEFAULT_none},
      {FromTy::Private, ToTy::OMP_DEFAULT_private},
      {FromTy::Shared, ToTy::OMP_DEFAULT_shared},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

llvm::omp::ProcBindKind vnoc(clause::ProcBind::AffinityPolicy T) {
  using FromTy = clause::ProcBind::AffinityPolicy;
  using ToTy = llvm::omp::ProcBindKind;

  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Close, ToTy::OMP_PROC_BIND_close},
      {FromTy::Master, ToTy::OMP_PROC_BIND_master},
      {FromTy::Primary, ToTy::OMP_PROC_BIND_primary},
      {FromTy::Spread, ToTy::OMP_PROC_BIND_spread},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPAtClauseKind vnoc(clause::At::ActionTime T) {
  using FromTy = clause::At::ActionTime;
  using ToTy = OpenMPAtClauseKind;

  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Compilation, OMPC_AT_compilation},
      {FromTy::Execution, OMPC_AT_execution},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPAtomicDefaultMemOrderClauseKind vnoc(MemoryOrder T) {
  using FromTy = MemoryOrder;
  using ToTy = OpenMPAtomicDefaultMemOrderClauseKind;

  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::AcqRel, OMPC_ATOMIC_DEFAULT_MEM_ORDER_acq_rel},
      {FromTy::Relaxed, OMPC_ATOMIC_DEFAULT_MEM_ORDER_relaxed},
      {FromTy::SeqCst, OMPC_ATOMIC_DEFAULT_MEM_ORDER_seq_cst},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPBindClauseKind vnoc(clause::Bind::Binding T) {
  using FromTy = clause::Bind::Binding;
  using ToTy = OpenMPBindClauseKind;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Parallel, OMPC_BIND_parallel},
      {FromTy::Teams, OMPC_BIND_teams},
      {FromTy::Thread, OMPC_BIND_thread},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPDefaultmapClauseKind vnoc(clause::Defaultmap::VariableCategory T) {
  using FromTy = clause::Defaultmap::VariableCategory;
  using ToTy = OpenMPDefaultmapClauseKind;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Aggregate, OMPC_DEFAULTMAP_aggregate},
      {FromTy::Pointer, OMPC_DEFAULTMAP_pointer},
      {FromTy::Scalar, OMPC_DEFAULTMAP_scalar},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPDefaultmapClauseModifier vnoc(clause::Defaultmap::ImplicitBehavior T) {
  using FromTy = clause::Defaultmap::ImplicitBehavior;
  using ToTy = OpenMPDefaultmapClauseModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Alloc, OMPC_DEFAULTMAP_MODIFIER_alloc},
      {FromTy::Default, OMPC_DEFAULTMAP_MODIFIER_default},
      {FromTy::Firstprivate, OMPC_DEFAULTMAP_MODIFIER_firstprivate},
      {FromTy::From, OMPC_DEFAULTMAP_MODIFIER_from},
      {FromTy::None, OMPC_DEFAULTMAP_MODIFIER_none},
      {FromTy::Present, OMPC_DEFAULTMAP_MODIFIER_present},
      {FromTy::To, OMPC_DEFAULTMAP_MODIFIER_to},
      {FromTy::Tofrom, OMPC_DEFAULTMAP_MODIFIER_tofrom},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPDeviceClauseModifier vnoc(clause::Device::DeviceModifier T) {
  using FromTy = clause::Device::DeviceModifier;
  using ToTy = OpenMPDeviceClauseModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Ancestor, OMPC_DEVICE_ancestor},
      {FromTy::DeviceNum, OMPC_DEVICE_device_num},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPDistScheduleClauseKind vnoc(clause::DistSchedule::Kind T) {
  using FromTy = clause::DistSchedule::Kind;
  using ToTy = OpenMPDistScheduleClauseKind;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Static, OMPC_DIST_SCHEDULE_static},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

#if 0
OpenMPGrainsizeClauseModifier vnoc(clause::Grainsize::Prescriptiveness T) {
  using FromTy = clause::Grainsize::Prescriptiveness;
  using ToTy = OpenMPGrainsizeClauseModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Strict, OMPC_GRAINSIZE_strict},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}
#endif

OpenMPLastprivateModifier vnoc(clause::Lastprivate::LastprivateModifier T) {
  using FromTy = clause::Lastprivate::LastprivateModifier;
  using ToTy = OpenMPLastprivateModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Conditional, OMPC_LASTPRIVATE_conditional},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPLinearClauseKind vnoc(clause::Linear::LinearModifier T) {
  using FromTy = clause::Linear::LinearModifier;
  using ToTy = OpenMPLinearClauseKind;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Ref, OMPC_LINEAR_ref},
      {FromTy::Uval, OMPC_LINEAR_uval},
      {FromTy::Val, OMPC_LINEAR_val},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPMapClauseKind vnoc(clause::Map::MapType T) {
  using FromTy = clause::Map::MapType;
  using ToTy = OpenMPMapClauseKind;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Alloc, OMPC_MAP_alloc},   {FromTy::To, OMPC_MAP_to},
      {FromTy::From, OMPC_MAP_from},     {FromTy::Tofrom, OMPC_MAP_tofrom},
      {FromTy::Delete, OMPC_MAP_delete}, {FromTy::Release, OMPC_MAP_release},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPMapModifierKind vnoc(clause::Map::MapTypeModifier T) {
  using FromTy = clause::Map::MapTypeModifier;
  using ToTy = OpenMPMapModifierKind;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Always, OMPC_MAP_MODIFIER_always},
      {FromTy::Close, OMPC_MAP_MODIFIER_close},
      {FromTy::Present, OMPC_MAP_MODIFIER_present},
      {FromTy::OmpxHold, OMPC_MAP_MODIFIER_ompx_hold},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

#if 0
OpenMPNumTasksClauseModifier vnoc(clause::NumTasks::Prescriptiveness T) {
  // OMPC_NUMTASKS_unknown -> std::nullopt
  using FromTy = clause::NumTasks::Prescriptiveness;
  using ToTy = OpenMPNumTasksClauseModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Strict, OMPC_NUMTASKS_strict},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}
#endif

OpenMPOrderClauseKind vnoc(clause::Order::Ordering T) {
  using FromTy = clause::Order::Ordering;
  using ToTy = OpenMPOrderClauseKind;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Concurrent, OMPC_ORDER_concurrent},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPOrderClauseModifier vnoc(clause::Order::OrderModifier T) {
  using FromTy = clause::Order::OrderModifier;
  using ToTy = OpenMPOrderClauseModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Reproducible, OMPC_ORDER_MODIFIER_reproducible},
      {FromTy::Unconstrained, OMPC_ORDER_MODIFIER_unconstrained},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPReductionClauseModifier vnoc(clause::Reduction::ReductionModifier T) {
  using FromTy = clause::Reduction::ReductionModifier;
  using ToTy = OpenMPReductionClauseModifier;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Default, OMPC_REDUCTION_default},
      {FromTy::Inscan, OMPC_REDUCTION_inscan},
      {FromTy::Task, OMPC_REDUCTION_task},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OpenMPSeverityClauseKind vnoc(clause::Severity::SevLevel T) {
  using FromTy = clause::Severity::SevLevel;
  using ToTy = OpenMPSeverityClauseKind;
  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Fatal, OMPC_SEVERITY_fatal},
      {FromTy::Warning, OMPC_SEVERITY_warning},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}

OverloadedOperatorKind vnoc(clause::DefinedOperator::IntrinsicOperator T) {
  using FromTy = clause::DefinedOperator::IntrinsicOperator;
  using ToTy = OverloadedOperatorKind;

  static const llvm::DenseMap<FromTy, ToTy> Conv = {
      {FromTy::Add, OO_Plus},      {FromTy::Subtract, OO_Minus},
      {FromTy::Multiply, OO_Star}, {FromTy::AND, OO_Amp},
      {FromTy::OR, OO_Pipe},       {FromTy::NEQV, OO_Caret},
      {FromTy::AND, OO_AmpAmp},    {FromTy::OR, OO_PipePipe},
  };
  if (auto F = Conv.find(T); F != Conv.end())
    return F->second;
  llvm_unreachable("Unexpected value");
}
} // namespace ext

namespace omp {
using tomp::makeList;

// "Convert" possibly null pointer to std::optional:
// if the pointer is null, return std::nullopt, otherwise return
// std::optional(ptr).
template <typename T> //
std::optional<std::remove_volatile_t<T> *> maybe(T *ptr) {
  if (ptr == nullptr)
    return std::nullopt;
  return ptr;
}

// Conditionally "convert" given value into std::optional<T>:
// if func(val) returns false, return std::nullopt, otherwise
// return std::optional(val).
// This is a macro to avoid evaluating "val" if the condition
// is false.
#define maybeIf(val, cond)                                                     \
  ((cond) ? std::optional<std::remove_reference_t<decltype(val)>>(val)         \
          : std::optional<std::remove_reference_t<decltype(val)>>{})

static const Expr *getPointerFromOffsetOp(const BinaryOperator *B) {
  BinaryOperatorKind Opc = B->getOpcode();
  assert(Opc == BO_Add || Opc == BO_Sub);
  if (B->getLHS()->getType()->isPointerType())
    return B->getLHS();
  assert(B->getRHS()->getType()->isPointerType());
  return B->getRHS();
}

// Taken from SemaOpenMP.cpp
static const Expr *getExprAsWritten(const Expr *E) {
  if (const auto *FE = dyn_cast<FullExpr>(E))
    E = FE->getSubExpr();

  if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
    E = MTE->getSubExpr();

  while (const auto *Binder = dyn_cast<CXXBindTemporaryExpr>(E))
    E = Binder->getSubExpr();

  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(E))
    E = ICE->getSubExprAsWritten();
  return E->IgnoreParens();
}

// Taken from SemaOpenMP.cpp
static const ValueDecl *getCanonicalDecl(const ValueDecl *D) {
  if (const auto *CED = dyn_cast<OMPCapturedExprDecl>(D))
    if (const auto *ME = dyn_cast<MemberExpr>(getExprAsWritten(CED->getInit())))
      D = ME->getMemberDecl();

  D = cast<ValueDecl>(D->getCanonicalDecl());
  return D;
}

Object makeObject(const Decl *D, const Expr *E = nullptr) {
  assert(D != nullptr);
  return Object{D, E};
}

Object makeObject(const Expr *E) {
  if (E->containsErrors())
    return Object{};
  if (auto *D = llvm::dyn_cast<DeclRefExpr>(E))
    return Object{getCanonicalDecl(D->getDecl()), E}; // ValueDecl
  if (auto *D = llvm::dyn_cast<MemberExpr>(E))
    return Object{getCanonicalDecl(D->getMemberDecl()), E}; // ValueDecl

  if (auto *A = llvm::dyn_cast<ArraySubscriptExpr>(E))
    return Object{makeObject(A->getBase()).id(), E};
  if (auto *A = llvm::dyn_cast<ArraySectionExpr>(E))
    return Object{makeObject(A->getBase()).id(), E};
  if (auto *A = llvm::dyn_cast<OMPArrayShapingExpr>(E))
    return Object{makeObject(A->getBase()).id(), E};

  if (auto *P = llvm::dyn_cast<ParenExpr>(E))
    return makeObject(P->getSubExpr());
  if (auto *P = llvm::dyn_cast<ImplicitCastExpr>(E))
    return makeObject(P->getSubExpr());
  if (auto *P = llvm::dyn_cast<OpaqueValueExpr>(E))
    return makeObject(P->getSourceExpr());

  if (auto *P = llvm::dyn_cast<BinaryOperator>(E)) {
    if (P->isAssignmentOp())
      return makeObject(P->getLHS());
    BinaryOperatorKind Opc = P->getOpcode();
    if (Opc == BO_Add || Opc == BO_Sub)
      return Object{makeObject(getPointerFromOffsetOp(P)).id(), E};
  }
  if (auto *P = llvm::dyn_cast<UnaryOperator>(E)) {
    UnaryOperatorKind Opc = P->getOpcode();
    if (Opc == UO_Deref || Opc == UO_AddrOf)
      return Object{makeObject(P->getSubExpr()).id(), E};
  }

  if (auto *P = llvm::dyn_cast<UnresolvedLookupExpr>(E))
    return Object{};
  if (auto *P = llvm::dyn_cast<CXXThisExpr>(E))
    return Object{P, E};

  E->dump();
  llvm_unreachable("Expecting DeclRefExpr");
}

ObjectList makeObjects(llvm::ArrayRef<const Expr *> Vars) {
  return makeList(Vars, [](const Expr *E) { return makeObject(E); });
}

clause::DefinedOperator makeDefinedOperator(OverloadedOperatorKind OOK) {
  // From the OpenMP 5.2 spec:
  // A reduction identifier is either an id-expression or one of the following
  // operators: +, - (deprecated), *, &, |, ^, && and ||.
  // Min and max are included in the implicitly-declared reduction operators.
  return clause::DefinedOperator{ext::conv(OOK)};
}

List<clause::ReductionOperator>
makeReductionOperators(const DeclarationName &RedId,
                       llvm::ArrayRef<const Expr *> Ops) {
  using ReductionOperator = clause::ReductionOperator;
  assert(!RedId.isDependentName());
  List<clause::ReductionOperator> RedOps;

  DeclarationName::NameKind Kind = RedId.getNameKind();
  if (Kind == DeclarationName::CXXOperatorName) {
    OverloadedOperatorKind OOK = RedId.getCXXOverloadedOperator();
    RedOps.push_back(ReductionOperator{makeDefinedOperator(OOK)});
    return RedOps;
  }

  assert(Kind == DeclarationName::Identifier);
  const auto *II = RedId.getAsIdentifierInfo();
  if (II->getName() == "min") {
    auto MinOp = clause::DefinedOperator::IntrinsicOperator::Min;
    RedOps.push_back(ReductionOperator{clause::DefinedOperator{MinOp}});
    return RedOps;
  }
  if (II->getName() == "max") {
    auto MaxOp = clause::DefinedOperator::IntrinsicOperator::Max;
    RedOps.push_back(ReductionOperator{clause::DefinedOperator{MaxOp}});
    return RedOps;
  }

  for (const Expr *Op : Ops) {
    if (Op == nullptr) {
      // This can happen in an invalid code.
      Object invalid{};
      RedOps.clear();
      RedOps.push_back(ReductionOperator{clause::ProcedureDesignator{invalid}});
      return RedOps;
    }

    // If it's not an intrinsic operator, then it should be a function call.
    auto *CallOp = cast<CallExpr>(Op);
    Object func = makeObject(CallOp->getCallee());
    RedOps.push_back(ReductionOperator{clause::ProcedureDesignator{func}});
  }

  return RedOps;
}

std::optional<Iterator> tryMakeIterator(const Expr *E) {
  if (E == nullptr)
    return std::nullopt;

  Iterator Iter;
  const auto &IterExpr = *llvm::cast<OMPIteratorExpr>(E);

  using IteratorSpecifier =
      tomp::type::IteratorSpecifierT<TypeTy, IdentTy, ExprTy>;
  using Range = tomp::type::RangeT<ExprTy>;

  for (int i = 0, e = IterExpr.numOfIterators(); i != e; ++i) {
    auto *ID = const_cast<Decl *>(IterExpr.getIteratorDecl(i));
    const OMPIteratorExpr::IteratorRange &IR = IterExpr.getIteratorRange(i);
    QualType Type = llvm::cast<VarDecl>(ID)->getType();
    Range R{{IR.Begin, IR.End, maybe(IR.Step)}};
    Iter.emplace_back(IteratorSpecifier{{Type, Object{ID, nullptr}, R}});
  }

  return Iter;
}

template <typename IteratorTy>
List<Mapper> makeMappers(llvm::iterator_range<IteratorTy> &&Range) {
  List<Mapper> Mappers;
  if (Range.empty())
    return Mappers;

  for (const Expr *M : Range) {
    if (M == nullptr || isa<UnresolvedLookupExpr>(M))
      continue;
    Object obj = makeObject(M);
    if (obj.id().index() != 0)
      Mappers.push_back(Mapper{/*MapperIdentifier=*/obj});
  }
  return Mappers;
}

clause::DependenceType makeDepType(OpenMPDependClauseKind DepKind) {
  switch (DepKind) {
  case OMPC_DEPEND_depobj:
    return clause::DependenceType::Depobj;
  case OMPC_DEPEND_in:
    return clause::DependenceType::In;
  case OMPC_DEPEND_inout:
    return clause::DependenceType::Inout;
  case OMPC_DEPEND_inoutset:
    return clause::DependenceType::Inoutset;
  case OMPC_DEPEND_mutexinoutset:
    return clause::DependenceType::Mutexinoutset;
  case OMPC_DEPEND_out:
    return clause::DependenceType::Out;
  case OMPC_DEPEND_sink:
    return clause::DependenceType::Sink;
  case OMPC_DEPEND_source:
    return clause::DependenceType::Source;
  default:
    llvm_unreachable("Unexpected OpenMPDependClauseKind");
  }
}

namespace clause {
MAKE_EMPTY_CLASS(AcqRel, OMPAcqRelClause);
MAKE_EMPTY_CLASS(Acquire, OMPAcquireClause);
MAKE_EMPTY_CLASS(Capture, OMPCaptureClause);
MAKE_EMPTY_CLASS(Compare, OMPCompareClause);
MAKE_EMPTY_CLASS(DynamicAllocators, OMPDynamicAllocatorsClause);
MAKE_EMPTY_CLASS(Full, OMPFullClause);
// MAKE_EMPTY_CLASS(Inbranch, OMPInbranchClause);
MAKE_EMPTY_CLASS(Mergeable, OMPMergeableClause);
MAKE_EMPTY_CLASS(Nogroup, OMPNogroupClause);
MAKE_EMPTY_CLASS(NoOpenmp, OMPNoOpenMPClause);
MAKE_EMPTY_CLASS(NoOpenmpRoutines, OMPNoOpenMPRoutinesClause);
MAKE_EMPTY_CLASS(NoParallelism, OMPNoParallelismClause);
// MAKE_EMPTY_CLASS(Notinbranch, OMPNotinbranchClause);
MAKE_EMPTY_CLASS(Nowait, OMPNowaitClause);
MAKE_EMPTY_CLASS(OmpxAttribute, OMPXAttributeClause);
MAKE_EMPTY_CLASS(OmpxBare, OMPXBareClause);
MAKE_EMPTY_CLASS(Read, OMPReadClause);
MAKE_EMPTY_CLASS(Relaxed, OMPRelaxedClause);
MAKE_EMPTY_CLASS(Release, OMPReleaseClause);
MAKE_EMPTY_CLASS(ReverseOffload, OMPReverseOffloadClause);
MAKE_EMPTY_CLASS(SeqCst, OMPSeqCstClause);
MAKE_EMPTY_CLASS(Simd, OMPSIMDClause);
MAKE_EMPTY_CLASS(Threads, OMPThreadsClause);
MAKE_EMPTY_CLASS(UnifiedAddress, OMPUnifiedAddressClause);
MAKE_EMPTY_CLASS(UnifiedSharedMemory, OMPUnifiedSharedMemoryClause);
// MAKE_EMPTY_CLASS(Unknown, OMPUnknownClause);
MAKE_EMPTY_CLASS(Untied, OMPUntiedClause);
MAKE_EMPTY_CLASS(Weak, OMPWeakClause);
MAKE_EMPTY_CLASS(Write, OMPWriteClause);

// Make artificial clauses
MAKE_EMPTY_CLASS(Depobj, OMPDepobjClause);
MAKE_EMPTY_CLASS(Flush, OMPFlushClause);

// Other clauses

Absent make(const OMPAbsentClause &C) {
  ArrayRef<OpenMPDirectiveKind> Kinds = C.getDirectiveKinds();
  return Absent{/*List=*/{Kinds.begin(), Kinds.end()}};
}

// AdjustArgs

Affinity make(const OMPAffinityClause &C) {
  return Affinity{
      {/*Iterator=*/std::nullopt,
       /*LocatorList=*/makeObjects(C.getVarRefs())},
  };
}

Align make(const OMPAlignClause &C) {
  return Align{/*Alignment=*/C.getAlignment()};
}

Aligned make(const OMPAlignedClause &C) {
  return Aligned{
      {/*Alignment=*/C.getAlignment(),
       /*List=*/makeObjects(C.getVarRefs())},
  };
}

Allocate make(const OMPAllocateClause &C) {
  return Allocate{
      {/*AllocatorComplexModifier=*/Allocator{C.getAllocator()},
       /*AlignModifier=*/std::nullopt,
       /*List=*/makeObjects(C.getVarRefs())},
  };
}

Allocator make(const OMPAllocatorClause &C) {
  return Allocator{/*Allocator=*/C.getAllocator()};
}

// AppendArgs

At make(const OMPAtClause &C) {
  return At{/*ActionTime=*/*ext::conv(C.getAtKind())};
}

AtomicDefaultMemOrder make(const OMPAtomicDefaultMemOrderClause &C) {
  return AtomicDefaultMemOrder{
      /*MemoryOrder=*/*ext::conv(C.getAtomicDefaultMemOrderKind())};
}

Bind make(const OMPBindClause &C) {
  return Bind{/*Binding=*/*ext::conv(C.getBindKind())};
}

Collapse make(const OMPCollapseClause &C) {
  return Collapse{/*N=*/C.getNumForLoops()};
}

Contains make(const OMPContainsClause &C) {
  ArrayRef<OpenMPDirectiveKind> Kinds = C.getDirectiveKinds();
  return Contains{/*List=*/{Kinds.begin(), Kinds.end()}};
}

Copyin make(const OMPCopyinClause &C) {
  return Copyin{/*List=*/makeObjects(C.getVarRefs())};
}

Copyprivate make(const OMPCopyprivateClause &C) {
  return Copyprivate{/*List=*/makeObjects(C.getVarRefs())};
}

Default make(const OMPDefaultClause &C) {
  return Default{/*DataSharingAttribute=*/*ext::conv(C.getDefaultKind())};
}

Defaultmap make(const OMPDefaultmapClause &C) {
  return Defaultmap{
      {
          /*ImplicitBehavior=*/*ext::conv(C.getDefaultmapModifier()),
          /*VariableCategory=*/ext::conv(C.getDefaultmapKind()),
      },
  };
}

Doacross make(DependenceType DepType,
              llvm::ArrayRef<const Expr *> Vars);

Depend make(const OMPDependClause &C) {
  std::optional<Iterator> maybeIterator = tryMakeIterator(C.getModifier());
  using TaskDep = Depend::TaskDep;

  OpenMPDependClauseKind DepKind = C.getDependencyKind();
  if (DepKind == OMPC_DEPEND_outallmemory) {
    return Depend{
        TaskDep{{/*DependenceType=*/DependenceType::Out,
                 /*Iterator=*/std::move(maybeIterator),
                 /*LocatorList=*/{}}},
    };
  }
  if (DepKind == OMPC_DEPEND_inoutallmemory) {
    return Depend{
        TaskDep{{/*DependenceType=*/DependenceType::Inout,
                 /*Iterator=*/std::move(maybeIterator),
                 /*LocatorList=*/{}}},
    };
  }
  if (DepKind != OMPC_DEPEND_source && DepKind != OMPC_DEPEND_sink) {
    return Depend{
        TaskDep{{/*DependenceType=*/makeDepType(DepKind),
                 /*Iterator=*/std::move(maybeIterator),
                 /*LocatorList=*/makeObjects(C.getVarRefs())}},
    };
  }

  // XXX The distance vectors are unavailable: they are in DSAStack, which is
  // a private member of Sema without an accessor.
  return Depend{make(makeDepType(DepKind), C.getVarRefs())};
}

Destroy make(const OMPDestroyClause &C) {
  if (Expr *Var = C.getInteropVar())
    return Destroy{/*DestroyVar=*/makeObject(Var)};
  return Destroy{/*DestroyVar=*/std::nullopt};
}

Detach make(const OMPDetachClause &C) {
  return Detach{/*EventHandle=*/makeObject(C.getEventHandler())};
}

Device make(const OMPDeviceClause &C) {
  OpenMPDeviceClauseModifier Modifier = C.getModifier();
  return Device{
      {/*DeviceModifier=*/ext::conv(Modifier),
       /*DeviceDescription=*/C.getDevice()},
  };
}

// DeviceType

DistSchedule make(const OMPDistScheduleClause &C) {
  return DistSchedule{
      {/*Kind=*/*ext::conv(C.getDistScheduleKind()),
       /*ChunkSize=*/maybe(C.getChunkSize())},
  };
}

Doacross make(DependenceType DepType,
              llvm::ArrayRef<const Expr *> Vars) {
  // XXX The loop iteration distance vector is unavailable (it's not exported
  // from Sema/SemaOpenMP.
  assert(DepType == DependenceType::Sink ||
         DepType == DependenceType::Source);
  return Doacross{{/*DependenceType=*/DepType, /*Vector=*/{}}};
}

Doacross make(const OMPDoacrossClause &C) {
  CLAUSET_ENUM_CONVERT( //
      convert, OpenMPDoacrossClauseModifier, DependenceType,
      // clang-format off
      MU(OMPC_DOACROSS_source, Source)
      MU(OMPC_DOACROSS_source_omp_cur_iteration, Source)
      MU(OMPC_DOACROSS_sink, Sink)
      MU(OMPC_DOACROSS_sink_omp_cur_iteration, Sink)
      // clang-format on
  );

  OpenMPDoacrossClauseModifier DepType = C.getDependenceType();
  if (DepType == OMPC_DOACROSS_source_omp_cur_iteration ||
      DepType == OMPC_DOACROSS_sink_omp_cur_iteration)
    return make(convert(DepType), {});

  return make(convert(DepType), C.getVarRefs());
}

// Enter

Exclusive make(const OMPExclusiveClause &C) {
  return Exclusive{/*List=*/makeObjects(C.getVarRefs())};
}

Fail make(const OMPFailClause &C) {
  CLAUSET_ENUM_CONVERT( //
      convert, llvm::omp::Clause, tomp::type::MemoryOrder,
      // clang-format off
      MS(OMPC_acq_rel, AcqRel)
      MS(OMPC_acquire, Acquire)
      MS(OMPC_relaxed, Relaxed)
      MS(OMPC_release, Release)
      MS(OMPC_seq_cst, SeqCst)
      // clang-format on
  );

  return Fail{/*MemoryOrder=*/convert(C.getFailParameter())};
}

Filter make(const OMPFilterClause &C) {
  return Filter{/*ThreadNum=*/C.getThreadID()};
}

Final make(const OMPFinalClause &C) {
  return Final{/*Finalize=*/C.getCondition()};
}

Firstprivate make(const OMPFirstprivateClause &C) {
  return Firstprivate{/*List=*/makeObjects(C.getVarRefs())};
}

From make(const OMPFromClause &C) {
  std::optional<tomp::type::MotionExpectation> maybeExp;
  if (llvm::is_contained(C.getMotionModifiers(), OMPC_MOTION_MODIFIER_present))
    maybeExp = tomp::type::MotionExpectation::Present;

  List<Mapper> Mappers = makeMappers(C.mapperlists());
  if (!Mappers.empty())
    assert(Mappers.size() == 1 || Mappers.size() == C.getVarRefs().size());

  return From{
      {/*Expectation=*/std::move(maybeExp),
       /*Mappers=*/maybeIf(Mappers, !Mappers.empty()),
       /*Iterator=*/std::nullopt,
       /*LocatorList=*/makeObjects(C.getVarRefs())},
  };
}

Grainsize make(const OMPGrainsizeClause &C) {
  return Grainsize{
      {/*Prescriptiveness=*/ext::conv(C.getModifier()),
       /*GrainSize=*/C.getGrainsize()},
  };
}

HasDeviceAddr make(const OMPHasDeviceAddrClause &C) {
  return HasDeviceAddr{/*List=*/makeObjects(C.getVarRefs())};
}

Hint make(const OMPHintClause &C) {
  return Hint{/*HintExpr=*/C.getHint()}; //
}

Holds make(const OMPHoldsClause &C) { return Holds{/*E=*/C.getExpr()}; }

If make(const OMPIfClause &C) {
  return If{
      {/*DirectiveNameModifier=*/ext::conv(C.getNameModifier()),
       /*IfExpression=*/C.getCondition()},
  };
}

Inclusive make(const OMPInclusiveClause &C) {
  return Inclusive{/*List=*/makeObjects(C.getVarRefs())};
}

// Indirect

Init make(const OMPInitClause &C) {
  const Expr *Var = C.getInteropVar();
  Init::InteropTypes Types(2);
  if (C.getIsTarget())
    Types.push_back(Init::InteropType::Target);
  if (C.getIsTargetSync())
    Types.push_back(Init::InteropType::Targetsync);
  auto Range = C.prefs();
  llvm::ArrayRef<const Expr *> Prefs(Range.begin(), Range.end());

  using T = decltype(Init::t);
  return Init{
      T{/*InteropPreference=*/maybeIf(Prefs, !Prefs.empty()),
        /*InteropTypes=*/Types,
        /*InteropVar=*/makeObject(Var)},
  };
}

// Initializer

InReduction make(const OMPInReductionClause &C) {
  auto OpsRange = C.reduction_ops();
  List<ReductionOperator> RedOps = makeReductionOperators(
      C.getNameInfo().getName(), {OpsRange.begin(), OpsRange.end()});
  assert(RedOps.size() == 1 || RedOps.size() == C.getVarRefs().size());

  return InReduction{
      {/*ReductionIdentifiers=*/RedOps,
       /*List=*/makeObjects(C.getVarRefs())},
  };
}

IsDevicePtr make(const OMPIsDevicePtrClause &C) {
  return IsDevicePtr{/*List=*/makeObjects(C.getVarRefs())};
}

Lastprivate make(const OMPLastprivateClause &C) {
  return Lastprivate{
      {/*LastprivateModifier=*/ext::conv(C.getKind()),
       /*List=*/makeObjects(C.getVarRefs())},
  };
}

Linear make(const OMPLinearClause &C) {
  return Linear{
      {/*StepSimpleModifier=*/std::nullopt,
       /*StepComplexModifier=*/C.getStep(),
       /*LinearModifier=*/*ext::conv(C.getModifier()),
       /*List=*/makeObjects(C.getVarRefs())},
  };
}

// Link

Map make(const OMPMapClause &C) {
  OpenMPMapClauseKind MapType = C.getMapType();
  List<Mapper> Mappers = makeMappers(C.mapperlists());
  if (!Mappers.empty())
    assert(Mappers.size() == 1 || Mappers.size() == C.getVarRefs().size());
  std::optional<Iterator> maybeIterator =
      tryMakeIterator(C.getIteratorModifier());

  Map::MapTypeModifiers Modifiers;
  auto &&KnownModifiers = llvm::make_filter_range(
      C.getMapTypeModifiers(), [](OpenMPMapModifierKind M) {
        // Skip "mapper" and "iterator" modifiers, because they're already
        // handled.
        return M != OMPC_MAP_MODIFIER_unknown &&
               M != OMPC_MAP_MODIFIER_mapper && M != OMPC_MAP_MODIFIER_iterator;
      });
  llvm::transform(KnownModifiers, std::back_inserter(Modifiers),
                  [](OpenMPMapModifierKind M) { return *ext::conv(M); });

  return Map{
      {/*MapType=*/ext::conv(MapType),
       /*MapTypeModifiers=*/
       maybeIf(Modifiers, !Modifiers.empty()),
       /*Mappers=*/maybeIf(Mappers, !Mappers.empty()),
       /*Iterator=*/std::move(maybeIterator),
       /*LocatorList=*/makeObjects(C.getVarRefs())},
  };
}

// Match

Message make(const OMPMessageClause &C) {
  return Message{/*MsgString=*/C.getMessageString()};
}

Nocontext make(const OMPNocontextClause &C) {
  return Nocontext{/*DoNotUpdateContext=*/C.getCondition()};
}

Nontemporal make(const OMPNontemporalClause &C) {
  return Nontemporal{/*List=*/makeObjects(C.getVarRefs())};
}

Novariants make(const OMPNovariantsClause &C) {
  return Novariants{/*DoNotUseVariant=*/C.getCondition()};
}

NumTasks make(const OMPNumTasksClause &C) {
  using T = decltype(NumTasks::t);
  return NumTasks{
      T{/*Prescriptiveness=*/ext::conv(C.getModifier()),
        /*NumTasks=*/C.getNumTasks()},
  };
}

NumTeams make(const OMPNumTeamsClause &C) {
  ArrayRef<const Expr *> UpperBounds = C.getVarRefs();
  List<NumTeams::Range> Ranges;
  for (const Expr *E : UpperBounds) {
    Ranges.push_back(
        {{/*LowerBound=*/std::nullopt, /*UpperBound=*/const_cast<Expr *>(E)}});
  }
  return NumTeams{std::move(Ranges)};
}

NumThreads make(const OMPNumThreadsClause &C) {
  return NumThreads{/*Nthreads=*/C.getNumThreads()};
}

OmpxDynCgroupMem make(const OMPXDynCGroupMemClause &C) {
  return OmpxDynCgroupMem{C.getSize()};
}

Order make(const OMPOrderClause &C) {
  return Order{
      {/*OrderModifier=*/ext::conv(C.getModifier()),
       /*Ordering=*/*ext::conv(C.getKind())},
  };
}

Ordered make(const OMPOrderedClause &C) {
  return Ordered{/*N=*/C.getNumForLoops()};
}

// Otherwise

Partial make(const OMPPartialClause &C) {
  return Partial{/*UnrollFactor=*/C.getFactor()};
}

Permutation make(const OMPPermutationClause &C) {
  Permutation::ArgList Args;
  for (const Expr *E : C.getArgsRefs())
    Args.push_back(const_cast<Expr *>(E));
  return Permutation{/*ArgList=*/std::move(Args)};
}

Priority make(const OMPPriorityClause &C) {
  return Priority{/*PriorityValue=*/C.getPriority()};
}

Private make(const OMPPrivateClause &C) {
  return Private{/*List=*/makeObjects(C.getVarRefs())};
}

ProcBind make(const OMPProcBindClause &C) {
  return ProcBind{/*AffinityPolicy=*/*ext::conv(C.getProcBindKind())};
}

Reduction make(const OMPReductionClause &C) {
  auto OpsRange = C.reduction_ops();
  List<ReductionOperator> RedOps = makeReductionOperators(
      C.getNameInfo().getName(), {OpsRange.begin(), OpsRange.end()});
  assert(RedOps.size() == 1 || RedOps.size() == C.getVarRefs().size());

  return Reduction{
      {/*ReductionModifier=*/ext::conv(C.getModifier()),
       /*ReductionIdentifiers=*/RedOps,
       /*List=*/makeObjects(C.getVarRefs())},
  };
}

Safelen make(const OMPSafelenClause &C) {
  return Safelen{/*Length=*/C.getSafelen()};
}

Schedule make(const OMPScheduleClause &C) {
  CLAUSET_ENUM_CONVERT( //
      convertKind, OpenMPScheduleClauseKind, Schedule::Kind,
      // clang-format off
      MU(OMPC_SCHEDULE_static, Static)
      MU(OMPC_SCHEDULE_dynamic, Dynamic)
      MU(OMPC_SCHEDULE_guided, Guided)
      MU(OMPC_SCHEDULE_auto, Auto)
      MU(OMPC_SCHEDULE_runtime, Runtime)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convertOrdMod, OpenMPScheduleClauseModifier, Schedule::OrderingModifier,
      // clang-format off
      MU(OMPC_SCHEDULE_MODIFIER_monotonic, Monotonic)
      MU(OMPC_SCHEDULE_MODIFIER_nonmonotonic, Nonmonotonic)
      // clang-format on
  );

  CLAUSET_ENUM_CONVERT( //
      convertChkMod, OpenMPScheduleClauseModifier, Schedule::ChunkModifier,
      // clang-format off
      MU(OMPC_SCHEDULE_MODIFIER_simd, Simd)
      // clang-format on
  );

  OpenMPScheduleClauseModifier OrdMod = C.getFirstScheduleModifier();
  OpenMPScheduleClauseModifier ChkMod = C.getSecondScheduleModifier();
  if (OrdMod == OMPC_SCHEDULE_MODIFIER_simd)
    std::swap(OrdMod, ChkMod);

  const Expr *ChkSize = C.getChunkSize();

  // clang-format off
  return Schedule{
      {/*Kind=*/convertKind(C.getScheduleKind()),
       /*OrderingModifier=*/maybeIf(convertOrdMod(OrdMod),
          OrdMod != OMPC_SCHEDULE_MODIFIER_unknown),
       /*ChunkModifier=*/maybeIf(convertChkMod(ChkMod),
          ChkMod != OMPC_SCHEDULE_MODIFIER_unknown),
       /*ChunkSize=*/maybeIf(ChkSize, ChkSize != nullptr)},
  };
  // clang-format on
}

Severity make(const OMPSeverityClause &C) {
  return Severity{/*SevLevel=*/*ext::conv(C.getSeverityKind())};
}

Shared make(const OMPSharedClause &C) {
  return Shared{/*List=*/makeObjects(C.getVarRefs())};
}

Simdlen make(const OMPSimdlenClause &C) {
  return Simdlen{/*Length=*/C.getSimdlen()};
}

Sizes make(const OMPSizesClause &C) {
  llvm::ArrayRef<const Expr *> S = C.getSizesRefs();
  return Sizes{/*SizeList=*/Sizes::SizeList(S.begin(), S.end())};
}

TaskReduction make(const OMPTaskReductionClause &C) {
  auto OpsRange = C.reduction_ops();
  List<ReductionOperator> RedOps = makeReductionOperators(
      C.getNameInfo().getName(), {OpsRange.begin(), OpsRange.end()});
  assert(RedOps.size() == 1 || RedOps.size() == C.getVarRefs().size());

  using T = decltype(TaskReduction::t);
  return TaskReduction{
      T{/*ReductionIdentifiers=*/RedOps,
        /*List=*/makeObjects(C.getVarRefs())},
  };
}

ThreadLimit make(const OMPThreadLimitClause &C) {
  /// XXX
  // return ThreadLimit{/*Threadlim=*/C.getThreadLimit()};
  return ThreadLimit{/*Threadlim=*/C.getVarRefs()[0]};
}

To make(const OMPToClause &C) {
  std::optional<tomp::type::MotionExpectation> maybeExp;
  if (llvm::is_contained(C.getMotionModifiers(), OMPC_MOTION_MODIFIER_present))
    maybeExp = tomp::type::MotionExpectation::Present;

  List<Mapper> Mappers = makeMappers(C.mapperlists());
  if (!Mappers.empty())
    assert(Mappers.size() == 1 || Mappers.size() == C.getVarRefs().size());

  return To{
      {/*Expectation=*/std::move(maybeExp),
       /*Mapper=*/maybeIf(Mappers, !Mappers.empty()),
       /*Iterator=*/std::nullopt,
       /*LocatorList=*/makeObjects(C.getVarRefs())},
  };
}

// Uniform

Update make(const OMPUpdateClause &C) {
  if (!C.isExtended())
    return Update{/*DependenceType=*/std::nullopt};

  CLAUSET_ENUM_CONVERT( //
      convert, OpenMPDependClauseKind, DependenceType,
      // clang-format off
      MU(OMPC_DEPEND_in, In)
      MU(OMPC_DEPEND_out, Out)
      MU(OMPC_DEPEND_inout, Inout)
      MU(OMPC_DEPEND_mutexinoutset, Mutexinoutset)
      MU(OMPC_DEPEND_inoutset, Inoutset)
      // OMPC_DEPEND_depobj is not allowed
      // clang-format on
  );

  OpenMPDependClauseKind DepType = C.getDependencyKind();
  return Update{/*DependenceType=*/maybeIf(convert(DepType),
                                           DepType != OMPC_DEPEND_unknown)};
}

Use make(const OMPUseClause &C) {
  return Use{/*InteropVar=*/makeObject(C.getInteropVar())};
}

UseDeviceAddr make(const OMPUseDeviceAddrClause &C) {
  return UseDeviceAddr{/*List=*/makeObjects(C.getVarRefs())};
}

UseDevicePtr make(const OMPUseDevicePtrClause &C) {
  return UseDevicePtr{/*List=*/makeObjects(C.getVarRefs())};
}

UsesAllocators make(const OMPUsesAllocatorsClause &C) {
  UsesAllocators::Allocators AllocList;
  AllocList.reserve(C.getNumberOfAllocators());

  using S = decltype(UsesAllocators::AllocatorSpec::t);
  for (int i = 0, e = C.getNumberOfAllocators(); i != e; ++i) {
    OMPUsesAllocatorsClause::Data Data = C.getAllocatorData(i);
    AllocList.push_back(UsesAllocators::AllocatorSpec{
        S{/*MemSpace=*/std::nullopt, /*TraitsArray=*/
          maybeIf(makeObject(Data.AllocatorTraits),
                  Data.AllocatorTraits != nullptr),
          /*Allocator=*/Data.Allocator}});
  }

  return UsesAllocators{/*Allocators=*/std::move(AllocList)};
}

// When

} // namespace clause

Clause makeClause(OMPClause *C) {
#define GEN_CLANG_CLAUSE_CLASS
#define CLAUSE_CLASS(Enum, Str, Class)                                         \
  if (auto S = llvm::dyn_cast<Class>(C))                                       \
    return Clause{/*Base class*/ {C->getClauseKind(), clause::make(*S)}, C};
#include "llvm/Frontend/OpenMP/OMP.inc"
  llvm_unreachable("Unexpected clause");
}

List<Clause> makeClauses(llvm::ArrayRef<OMPClause *> Clauses) {
  return makeList(Clauses, [](OMPClause *C) { return makeClause(C); });
}
} // namespace omp

namespace clang {
bool ExtConstructDecomposition::ExtConstruct::addClause(const omp::Clause &C) {
  if (llvm::omp::isUniqueClause(C.id)) {
    // If C is a unique clause, only add it if one like this isn't
    // already there.
    auto F = llvm::find_if(
        clauses, [Id = C.id](const omp::Clause &T) { return T.id == Id; });
    if (F != clauses.end())
      return false;
  } else {
    // Don't add multiple clauses that correspond to the same AST clause.
    if (C.tag.getFlags() == C.tag.Explicit && C.tag.getPointer()) {
      void *Ptr = C.tag.getPointer();
      auto F = llvm::find_if(clauses, [=](const omp::Clause &T) {
        return T.tag.getFlags() == T.tag.Explicit && T.tag.getPointer() == Ptr;
      });
      if (F != clauses.end())
        return false;
    }
  }
  clauses.push_back(C);
  return true;
}

ExtConstructDecomposition::ExtConstructDecomposition(
    OpenMPDirectiveKind DKind, llvm::ArrayRef<omp::Clause> Clauses,
    uint32_t OMPVersion)
    : InputDKind(DKind), ClauseStorage(Clauses.begin(), Clauses.end()) {

  Decomposer =
      std::make_unique<DecomposerTy>(OMPVersion, *this, DKind, ClauseStorage,
                                     /*makeImplicit=*/false);
  if (Decomposer->output.empty())
    return;

  llvm::SmallVector<OpenMPDirectiveKind> UnitKinds;
  std::ignore = llvm::omp::getLeafOrCompositeConstructs(DKind, UnitKinds);
  for (OpenMPDirectiveKind Unit : UnitKinds) {
    Output.push_back(ExtConstruct(Unit));
    for (OpenMPDirectiveKind Leaf : llvm::omp::getLeafConstructsOrSelf(Unit)) {
      auto P = CompositionMap.insert(std::make_pair(Leaf, Unit));
      assert(P.second && "Repeated leaf in compound directive");
    }
  }

  for (tomp::DirectiveWithClauses<omp::Clause> &DWC : Decomposer->output) {
    ExtConstruct &Con = getConstruct(DWC.id);
    for (const omp::Clause &C : DWC.clauses)
      Con.addClause(C);
  }
}

bool ExtConstructDecomposition::postApply(
    const omp::Clause &C, llvm::SmallVector<OpenMPDirectiveKind> *Modified) {
  llvm::SmallVector<size_t> ClauseCount;

  for (tomp::DirectiveWithClauses<omp::Clause> &DWC : Decomposer->output)
    ClauseCount.push_back(DWC.clauses.size());

  ClauseStorage.push_back(C);
  bool Applied = Decomposer->postApply(C);

  if (!Applied)
    return false;

  for (size_t I = 0, E = Decomposer->output.size(); I != E; ++I) {
    tomp::DirectiveWithClauses<omp::Clause> &DWC = Decomposer->output[I];
    size_t SizeThen = ClauseCount[I], SizeNow = DWC.clauses.size();
    if (SizeThen < SizeNow) {
      ExtConstruct &Con = getConstruct(DWC.id);
      for (size_t J = SizeThen; J != SizeNow; ++J)
        Con.addClause(DWC.clauses[J]);
      if (Modified)
        Modified->push_back(Con.DKind);
    }
  }
  return true;
}

ExtConstructDecomposition::ExtConstruct &
ExtConstructDecomposition::getConstruct(OpenMPDirectiveKind Leaf) {
  assert(llvm::omp::isLeafConstruct(Leaf) && "Expecting leaf construct");
  OpenMPDirectiveKind ConKind = CompositionMap.at(Leaf);
  auto F = llvm::find_if(
      Output, [=](const ExtConstruct &T) { return T.DKind == ConKind; });
  assert(F != Output.end() && "Constituent not found");
  return *F;
}

// Given an object, return its base object if one exists.
std::optional<omp::Object>
ExtConstructDecomposition::getBaseObject(const omp::Object &Obj) {
  return std::nullopt; // XXX
}

// Return the iteration variable of the associated loop if any.
std::optional<omp::Object> ExtConstructDecomposition::getLoopIterVar() {
  return std::nullopt; // XXX
}
} // namespace clang
