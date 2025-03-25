#ifndef CLANG_SEMA_SEMAOPENMPEXT_H
#define CLANG_SEMA_SEMAOPENMPEXT_H

#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/OpenMPKinds.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Frontend/OpenMP/ClauseT.h"
#include "llvm/Frontend/OpenMP/ConstructDecompositionT.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/MathExtras.h"

#include <iterator>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace clang {
class Decl;
class Expr;
class OMPClause;
} // namespace clang

namespace omp {
using TypeTy = clang::QualType;
using IdentTy = std::variant<std::monostate, const clang::Decl *,
                             const clang::CXXThisExpr *>;
using ExprTy = const clang::Expr *;
} // namespace omp

template <> struct tomp::type::ObjectT<omp::IdentTy, omp::ExprTy> {
  using IdType = omp::IdentTy;
  using ExprType = omp::ExprTy;

  IdType id() const { return declaration; }
  ExprType ref() const { return reference; }

  IdType declaration;
  ExprType reference;
};

namespace omp {
template <typename T> //
using List = tomp::type::ListT<T>;

using Object = tomp::type::ObjectT<IdentTy, ExprTy>;
using ObjectList = tomp::type::ObjectListT<IdentTy, ExprTy>;
using Iterator = tomp::type::IteratorT<TypeTy, IdentTy, ExprTy>;
using Range = tomp::type::RangeT<ExprTy>;
using Mapper = tomp::type::MapperT<IdentTy, ExprTy>;

namespace clause {
using DefinedOperator = tomp::type::DefinedOperatorT<IdentTy, ExprTy>;
using ProcedureDesignator = tomp::type::ProcedureDesignatorT<IdentTy, ExprTy>;
using ReductionOperator = tomp::type::ReductionIdentifierT<IdentTy, ExprTy>;
using DependenceType = tomp::type::DependenceType;
using Prescriptiveness = tomp::type::Prescriptiveness;

using Absent = tomp::clause::AbsentT<TypeTy, IdentTy, ExprTy>;
using AcqRel = tomp::clause::AcqRelT<TypeTy, IdentTy, ExprTy>;
using Acquire = tomp::clause::AcquireT<TypeTy, IdentTy, ExprTy>;
using AdjustArgs = tomp::clause::AdjustArgsT<TypeTy, IdentTy, ExprTy>;
using Affinity = tomp::clause::AffinityT<TypeTy, IdentTy, ExprTy>;
using Align = tomp::clause::AlignT<TypeTy, IdentTy, ExprTy>;
using Aligned = tomp::clause::AlignedT<TypeTy, IdentTy, ExprTy>;
using Allocate = tomp::clause::AllocateT<TypeTy, IdentTy, ExprTy>;
using Allocator = tomp::clause::AllocatorT<TypeTy, IdentTy, ExprTy>;
using AppendArgs = tomp::clause::AppendArgsT<TypeTy, IdentTy, ExprTy>;
using At = tomp::clause::AtT<TypeTy, IdentTy, ExprTy>;
using AtomicDefaultMemOrder =
    tomp::clause::AtomicDefaultMemOrderT<TypeTy, IdentTy, ExprTy>;
using Bind = tomp::clause::BindT<TypeTy, IdentTy, ExprTy>;
using Capture = tomp::clause::CaptureT<TypeTy, IdentTy, ExprTy>;
using Collapse = tomp::clause::CollapseT<TypeTy, IdentTy, ExprTy>;
using Compare = tomp::clause::CompareT<TypeTy, IdentTy, ExprTy>;
using Contains = tomp::clause::ContainsT<TypeTy, IdentTy, ExprTy>;
using Copyin = tomp::clause::CopyinT<TypeTy, IdentTy, ExprTy>;
using Copyprivate = tomp::clause::CopyprivateT<TypeTy, IdentTy, ExprTy>;
using Default = tomp::clause::DefaultT<TypeTy, IdentTy, ExprTy>;
using Defaultmap = tomp::clause::DefaultmapT<TypeTy, IdentTy, ExprTy>;
using Depend = tomp::clause::DependT<TypeTy, IdentTy, ExprTy>;
using Destroy = tomp::clause::DestroyT<TypeTy, IdentTy, ExprTy>;
using Detach = tomp::clause::DetachT<TypeTy, IdentTy, ExprTy>;
using Device = tomp::clause::DeviceT<TypeTy, IdentTy, ExprTy>;
using DeviceType = tomp::clause::DeviceTypeT<TypeTy, IdentTy, ExprTy>;
using DistSchedule = tomp::clause::DistScheduleT<TypeTy, IdentTy, ExprTy>;
using Doacross = tomp::clause::DoacrossT<TypeTy, IdentTy, ExprTy>;
using DynamicAllocators =
    tomp::clause::DynamicAllocatorsT<TypeTy, IdentTy, ExprTy>;
using Enter = tomp::clause::EnterT<TypeTy, IdentTy, ExprTy>;
using Exclusive = tomp::clause::ExclusiveT<TypeTy, IdentTy, ExprTy>;
using Fail = tomp::clause::FailT<TypeTy, IdentTy, ExprTy>;
using Filter = tomp::clause::FilterT<TypeTy, IdentTy, ExprTy>;
using Final = tomp::clause::FinalT<TypeTy, IdentTy, ExprTy>;
using Firstprivate = tomp::clause::FirstprivateT<TypeTy, IdentTy, ExprTy>;
using From = tomp::clause::FromT<TypeTy, IdentTy, ExprTy>;
using Full = tomp::clause::FullT<TypeTy, IdentTy, ExprTy>;
using Grainsize = tomp::clause::GrainsizeT<TypeTy, IdentTy, ExprTy>;
using HasDeviceAddr = tomp::clause::HasDeviceAddrT<TypeTy, IdentTy, ExprTy>;
using Hint = tomp::clause::HintT<TypeTy, IdentTy, ExprTy>;
using Holds = tomp::clause::HoldsT<TypeTy, IdentTy, ExprTy>;
using If = tomp::clause::IfT<TypeTy, IdentTy, ExprTy>;
using Inbranch = tomp::clause::InbranchT<TypeTy, IdentTy, ExprTy>;
using Inclusive = tomp::clause::InclusiveT<TypeTy, IdentTy, ExprTy>;
using Indirect = tomp::clause::IndirectT<TypeTy, IdentTy, ExprTy>;
using Init = tomp::clause::InitT<TypeTy, IdentTy, ExprTy>;
using Initializer = tomp::clause::InitializerT<TypeTy, IdentTy, ExprTy>;
using InReduction = tomp::clause::InReductionT<TypeTy, IdentTy, ExprTy>;
using IsDevicePtr = tomp::clause::IsDevicePtrT<TypeTy, IdentTy, ExprTy>;
using Lastprivate = tomp::clause::LastprivateT<TypeTy, IdentTy, ExprTy>;
using Linear = tomp::clause::LinearT<TypeTy, IdentTy, ExprTy>;
using Link = tomp::clause::LinkT<TypeTy, IdentTy, ExprTy>;
using Map = tomp::clause::MapT<TypeTy, IdentTy, ExprTy>;
using Match = tomp::clause::MatchT<TypeTy, IdentTy, ExprTy>;
using Mergeable = tomp::clause::MergeableT<TypeTy, IdentTy, ExprTy>;
using Message = tomp::clause::MessageT<TypeTy, IdentTy, ExprTy>;
using Nocontext = tomp::clause::NocontextT<TypeTy, IdentTy, ExprTy>;
using Nogroup = tomp::clause::NogroupT<TypeTy, IdentTy, ExprTy>;
using Nontemporal = tomp::clause::NontemporalT<TypeTy, IdentTy, ExprTy>;
using NoOpenmp = tomp::clause::NoOpenmpT<TypeTy, IdentTy, ExprTy>;
using NoOpenmpRoutines =
    tomp::clause::NoOpenmpRoutinesT<TypeTy, IdentTy, ExprTy>;
using NoParallelism = tomp::clause::NoParallelismT<TypeTy, IdentTy, ExprTy>;
using Notinbranch = tomp::clause::NotinbranchT<TypeTy, IdentTy, ExprTy>;
using Novariants = tomp::clause::NovariantsT<TypeTy, IdentTy, ExprTy>;
using Nowait = tomp::clause::NowaitT<TypeTy, IdentTy, ExprTy>;
using NumTasks = tomp::clause::NumTasksT<TypeTy, IdentTy, ExprTy>;
using NumTeams = tomp::clause::NumTeamsT<TypeTy, IdentTy, ExprTy>;
using NumThreads = tomp::clause::NumThreadsT<TypeTy, IdentTy, ExprTy>;
using OmpxAttribute = tomp::clause::OmpxAttributeT<TypeTy, IdentTy, ExprTy>;
using OmpxBare = tomp::clause::OmpxBareT<TypeTy, IdentTy, ExprTy>;
using OmpxDynCgroupMem =
    tomp::clause::OmpxDynCgroupMemT<TypeTy, IdentTy, ExprTy>;
using Order = tomp::clause::OrderT<TypeTy, IdentTy, ExprTy>;
using Ordered = tomp::clause::OrderedT<TypeTy, IdentTy, ExprTy>;
using Otherwise = tomp::clause::OtherwiseT<TypeTy, IdentTy, ExprTy>;
using Partial = tomp::clause::PartialT<TypeTy, IdentTy, ExprTy>;
using Permutation = tomp::clause::PermutationT<TypeTy, IdentTy, ExprTy>;
using Priority = tomp::clause::PriorityT<TypeTy, IdentTy, ExprTy>;
using Private = tomp::clause::PrivateT<TypeTy, IdentTy, ExprTy>;
using ProcBind = tomp::clause::ProcBindT<TypeTy, IdentTy, ExprTy>;
using Read = tomp::clause::ReadT<TypeTy, IdentTy, ExprTy>;
using Reduction = tomp::clause::ReductionT<TypeTy, IdentTy, ExprTy>;
using Relaxed = tomp::clause::RelaxedT<TypeTy, IdentTy, ExprTy>;
using Release = tomp::clause::ReleaseT<TypeTy, IdentTy, ExprTy>;
using ReverseOffload = tomp::clause::ReverseOffloadT<TypeTy, IdentTy, ExprTy>;
using Safelen = tomp::clause::SafelenT<TypeTy, IdentTy, ExprTy>;
using Schedule = tomp::clause::ScheduleT<TypeTy, IdentTy, ExprTy>;
using SeqCst = tomp::clause::SeqCstT<TypeTy, IdentTy, ExprTy>;
using Severity = tomp::clause::SeverityT<TypeTy, IdentTy, ExprTy>;
using Shared = tomp::clause::SharedT<TypeTy, IdentTy, ExprTy>;
using Simd = tomp::clause::SimdT<TypeTy, IdentTy, ExprTy>;
using Simdlen = tomp::clause::SimdlenT<TypeTy, IdentTy, ExprTy>;
using Sizes = tomp::clause::SizesT<TypeTy, IdentTy, ExprTy>;
using TaskReduction = tomp::clause::TaskReductionT<TypeTy, IdentTy, ExprTy>;
using ThreadLimit = tomp::clause::ThreadLimitT<TypeTy, IdentTy, ExprTy>;
using Threads = tomp::clause::ThreadsT<TypeTy, IdentTy, ExprTy>;
using To = tomp::clause::ToT<TypeTy, IdentTy, ExprTy>;
using UnifiedAddress = tomp::clause::UnifiedAddressT<TypeTy, IdentTy, ExprTy>;
using UnifiedSharedMemory =
    tomp::clause::UnifiedSharedMemoryT<TypeTy, IdentTy, ExprTy>;
using Uniform = tomp::clause::UniformT<TypeTy, IdentTy, ExprTy>;
using Unknown = tomp::clause::UnknownT<TypeTy, IdentTy, ExprTy>;
using Untied = tomp::clause::UntiedT<TypeTy, IdentTy, ExprTy>;
using Update = tomp::clause::UpdateT<TypeTy, IdentTy, ExprTy>;
using Use = tomp::clause::UseT<TypeTy, IdentTy, ExprTy>;
using UseDeviceAddr = tomp::clause::UseDeviceAddrT<TypeTy, IdentTy, ExprTy>;
using UseDevicePtr = tomp::clause::UseDevicePtrT<TypeTy, IdentTy, ExprTy>;
using UsesAllocators = tomp::clause::UsesAllocatorsT<TypeTy, IdentTy, ExprTy>;
using Weak = tomp::clause::WeakT<TypeTy, IdentTy, ExprTy>;
using When = tomp::clause::WhenT<TypeTy, IdentTy, ExprTy>;
using Write = tomp::clause::WriteT<TypeTy, IdentTy, ExprTy>;
} // namespace clause

using tomp::type::operator==;

// Manually define artificial clauses
struct Depobj {
  using EmptyTrait = std::true_type;
};
struct Flush {
  using EmptyTrait = std::true_type;
};

using ClauseBase = tomp::ClauseT<TypeTy, IdentTy, ExprTy,
                                 // Artificial clauses:
                                 Depobj, Flush>;

struct Clause : public ClauseBase {
  struct TagType;

  Clause(ClauseBase &&Base, TagType Tag)
      : ClauseBase(std::move(Base)), tag(Tag) {}
  Clause(ClauseBase &&Base, const clang::OMPClause *C = nullptr)
      : Clause(std::move(Base), TagType::get(C)) {}

  struct TagType {
    using StorageTy = std::common_type_t<uint64_t, uintptr_t>;

    enum : unsigned {
      Explicit = 0, // The clause should be treated as explicit. If the clause
                    // originated from AST, Storage contains the OMPClause*,
                    // otherwise nullptr.
                    // This is the value when TagType is default-initialized.
                    // Explicit clauses will be applied early (before captured
                    // region is closed), non-explicit clauses may be applied
                    // after closing of the region.
      Simple,       // The clause is implicit, and should be turned into an
                    // AST node with default settings (empty location, etc.).
                    // Storage is ignored.
      Mapping,      // The clause implicit and is "map", Storage is A[2]:
                    // A[0] = Defaultmap category, A[1] = MapType.
    };

    static TagType get(const clang::OMPClause *C) {
      return TagType{Explicit, reinterpret_cast<StorageTy>(C)};
    }
    static TagType get() { return TagType{Simple, 0x0}; }
    static TagType get(unsigned DefType, unsigned MapType) {
      return TagType{Mapping, llvm::Make_64(/*Hi=*/MapType, /*Lo=*/DefType)};
    }

    unsigned getFlags() const { return Flags; }
    void *getPointer() const { return reinterpret_cast<void *>(Storage); }
    std::pair<unsigned, unsigned> getMapKinds() const {
      return {llvm::Lo_32(Storage), llvm::Hi_32(Storage)};
    }

    unsigned Flags = 0x0;
    StorageTy Storage = 0x0;
  };

  TagType tag;
};

Clause makeClause(clang::OMPClause *C);
List<Clause> makeClauses(llvm::ArrayRef<clang::OMPClause *> AstClauses);

Object makeObject(const clang::Expr *E);
ObjectList makeObjects(llvm::ArrayRef<const clang::Expr *> Vars);
} // namespace omp

namespace ext {
// "Rename" the omp namespace to ext to make it visibly distinct from
// other OpenMP symbols.
using namespace omp;

using MemoryOrder = tomp::type::MemoryOrder;

inline std::optional<tomp::type::DirectiveName> conv(llvm::omp::Directive T) {
  if (T != llvm::omp::Directive::OMPD_unknown)
    return T;
  return std::nullopt;
}

// Conversions between enums used by omp::Clause and clang.
// Using trailing return, otherwise these declarations are visually unparseable.
auto conv(llvm::omp::DefaultKind T)
    -> std::optional<clause::Default::DataSharingAttribute>;
auto conv(llvm::omp::ProcBindKind T)
    -> std::optional<clause::ProcBind::AffinityPolicy>;
auto conv(clang::OpenMPAtClauseKind T) //
    -> std::optional<clause::At::ActionTime>;
auto conv(clang::OpenMPAtomicDefaultMemOrderClauseKind T)
    -> std::optional<MemoryOrder>;
auto conv(clang::OpenMPBindClauseKind T)
    -> std::optional<clause::Bind::Binding>;
auto conv(clang::OpenMPDefaultmapClauseKind T)
    -> std::optional<clause::Defaultmap::VariableCategory>;
auto conv(clang::OpenMPDefaultmapClauseModifier T)
    -> std::optional<clause::Defaultmap::ImplicitBehavior>;
auto conv(clang::OpenMPDeviceClauseModifier T)
    -> std::optional<clause::Device::DeviceModifier>;
auto conv(clang::OpenMPDistScheduleClauseKind T)
    -> std::optional<clause::DistSchedule::Kind>;
auto conv(clang::OpenMPGrainsizeClauseModifier T)
    -> std::optional<clause::Grainsize::Prescriptiveness>;
auto conv(clang::OpenMPLastprivateModifier T)
    -> std::optional<clause::Lastprivate::LastprivateModifier>;
auto conv(clang::OpenMPLinearClauseKind T)
    -> std::optional<clause::Linear::LinearModifier>;
auto conv(clang::OpenMPMapClauseKind T) //
    -> std::optional<clause::Map::MapType>;
auto conv(clang::OpenMPMapModifierKind T)
    -> std::optional<clause::Map::MapTypeModifier>;
auto conv(clang::OpenMPNumTasksClauseModifier T)
    -> std::optional<clause::NumTasks::Prescriptiveness>;
auto conv(clang::OpenMPOrderClauseKind T)
    -> std::optional<clause::Order::Ordering>;
auto conv(clang::OpenMPOrderClauseModifier T)
    -> std::optional<clause::Order::OrderModifier>;
auto conv(clang::OpenMPReductionClauseModifier T)
    -> std::optional<clause::Reduction::ReductionModifier>;
auto conv(clang::OpenMPSeverityClauseKind T)
    -> std::optional<clause::Severity::SevLevel>;
auto conv(clang::OverloadedOperatorKind T)
    -> clause::DefinedOperator::IntrinsicOperator;

auto vnoc(clause::Default::DataSharingAttribute T) -> llvm::omp::DefaultKind;
auto vnoc(clause::ProcBind::AffinityPolicy T) -> llvm::omp::ProcBindKind;
auto vnoc(clause::At::ActionTime T) -> clang::OpenMPAtClauseKind;
auto vnoc(MemoryOrder T) -> clang::OpenMPAtomicDefaultMemOrderClauseKind;
auto vnoc(clause::Bind::Binding T) -> clang::OpenMPBindClauseKind;
auto vnoc(clause::Defaultmap::VariableCategory T)
    -> clang::OpenMPDefaultmapClauseKind;
auto vnoc(clause::Defaultmap::ImplicitBehavior T)
    -> clang::OpenMPDefaultmapClauseModifier;
auto vnoc(clause::Device::DeviceModifier T)
    -> clang::OpenMPDeviceClauseModifier;
auto vnoc(clause::DistSchedule::Kind T) -> clang::OpenMPDistScheduleClauseKind;
//auto vnoc(clause::Grainsize::Prescriptiveness T)
//    -> clang::OpenMPGrainsizeClauseModifier;
auto vnoc(clause::Lastprivate::LastprivateModifier T)
    -> clang::OpenMPLastprivateModifier;
auto vnoc(clause::Linear::LinearModifier T) -> clang::OpenMPLinearClauseKind;
auto vnoc(clause::Map::MapType T) -> clang::OpenMPMapClauseKind;
auto vnoc(clause::Map::MapTypeModifier T) -> clang::OpenMPMapModifierKind;
//auto vnoc(clause::NumTasks::Prescriptiveness)
//    -> clang::OpenMPNumTasksClauseModifier;
auto vnoc(clause::Order::Ordering T) -> clang::OpenMPOrderClauseKind;
auto vnoc(clause::Order::OrderModifier T) -> clang::OpenMPOrderClauseModifier;
auto vnoc(clause::Reduction::ReductionModifier T)
    -> clang::OpenMPReductionClauseModifier;
auto vnoc(clause::Severity::SevLevel T) -> clang::OpenMPSeverityClauseKind;
} // namespace ext

namespace clang {
using OpenMPDirectiveKind = llvm::omp::Directive;
using OpenMPClauseKind = llvm::omp::Clause;

template <typename ContainerTy> struct QueueAdapter {
  using value_type = typename ContainerTy::value_type;
  using iterator = typename ContainerTy::iterator;
  using const_iterator = typename ContainerTy::const_iterator;

  QueueAdapter() : Container(nullptr), StartIdx(0) {}
  QueueAdapter(ContainerTy &C) : Container(&C), StartIdx(C.size()) {}
  template <typename OtherTy,
            std::enable_if_t<std::is_same_v<value_type, OtherTy::value_type>,
                             int> = 0>
  QueueAdapter &operator=(OtherTy &C) {
    Container = &C;
    StartIdx = C.size();
  }

  value_type &take() {
    assert(StartIdx < Container->size() && "Taking from empty queue");
    return (*Container)[StartIdx++];
  }

  ArrayRef<value_type> takeAll() {
    ArrayRef Res(begin(), end());
    StartIdx += Res.size();
    return Res;
  }

  size_t size() const { return Container->size() - StartIdx; }
  bool empty() const { return size() == 0; }

  ContainerTy *Container;

  iterator begin() {
    iterator it = Container->begin();
    std::advance(it, StartIdx);
    return it;
  }
  iterator end() { return Container->end(); }
  const iterator begin() const {
    const iterator it = Container->begin();
    std::advance(it, StartIdx);
    return it;
  }
  const_iterator end() const { return Container->end(); }

private:
  size_t StartIdx;
};

template <typename ContainerTy>
QueueAdapter(ContainerTy) -> QueueAdapter<ContainerTy>;

struct ExtConstructDecomposition {
  struct ExtConstruct : private tomp::DirectiveWithClauses<omp::Clause> {
    using BaseTy = tomp::DirectiveWithClauses<omp::Clause>;

    ExtConstruct(const ExtConstruct &Con)
        : BaseTy(Con), DKind(id), ClausesQ(clauses) {}
    ExtConstruct(ExtConstruct &&Con)
        : BaseTy(std::move(Con)), DKind(id), ClausesQ(clauses) {}
    ExtConstruct(OpenMPDirectiveKind DK = llvm::omp::OMPD_unknown)
        : DKind(id), ClausesQ(clauses) {
      id = DK;
    }
    bool addClause(const omp::Clause &C);

    OpenMPDirectiveKind &DKind;
    QueueAdapter<decltype(clauses)> ClausesQ;
  };

  ExtConstructDecomposition(OpenMPDirectiveKind DKind,
                            llvm::ArrayRef<omp::Clause> Clauses,
                            uint32_t OMPVersion);

  bool postApply(const omp::Clause &C,
                 llvm::SmallVector<OpenMPDirectiveKind> *Modified);

  // Given an object, return its base object if one exists.
  std::optional<omp::Object> getBaseObject(const omp::Object &Obj);

  // Return the iteration variable of the associated loop if any.
  std::optional<omp::Object> getLoopIterVar();

  OpenMPDirectiveKind InputDKind;
  llvm::SmallVector<omp::Clause> ClauseStorage;
  // Map: leaf -> leaf-or-composite construct containing the leaf,
  // for each leaf constituent of the original directive.
  llvm::DenseMap<OpenMPDirectiveKind, OpenMPDirectiveKind> CompositionMap;
  llvm::SmallVector<ExtConstruct> Output;

private:
  using DecomposerTy =
      tomp::ConstructDecompositionT<omp::Clause, ExtConstructDecomposition>;
  std::unique_ptr<DecomposerTy> Decomposer;

  ExtConstruct &getConstruct(OpenMPDirectiveKind Leaf);
};
} // namespace clang

#endif // CLANG_SEMA_SEMAOPENMPEXT_H
