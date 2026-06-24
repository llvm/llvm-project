//===- TUSummaryBuilder.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummary.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include <list>
#include <memory>
#include <type_traits>
#include <utility>

using namespace clang;
using namespace ssaf;

EntityId TUSummaryBuilder::addEntity(const EntityName &EN,
                                     EntityLinkageType Linkage) {
  EntityId Id = Summary.IdTable.getId(EN);
  [[maybe_unused]] EntityLinkageType Link =
      Summary.LinkageTable.try_emplace(Id, Linkage).first->second.getLinkage();
  // Even if we had in the past a linkage, that must bee the same as we set now.
  assert(Link == Linkage);
  return Id;
}

std::pair<EntitySummary *, bool>
TUSummaryBuilder::addSummaryImpl(EntityId Entity,
                                 std::unique_ptr<EntitySummary> &&Data) {
  auto &EntitySummaries = Summary.Data[Data->getSummaryName()];
  auto [It, Inserted] = EntitySummaries.try_emplace(Entity, std::move(Data));
  return {It->second.get(), Inserted};
}

namespace clang::ssaf::v2 {

template <typename... Headers> struct Header : Headers... {
  template <class... Ts>
  Header(Ts &&...Args) : Headers{std::forward<Ts>(Args)}... {}
};

struct Name {
  const llvm::StringRef Name;
};
struct Description {
  const llvm::StringRef Description;
};

template <typename Base, typename... CtorParamTypes> struct InstantiateOnHeap {
  static_assert(std::has_virtual_destructor_v<Base>);
  using ReturnType = std::unique_ptr<Base>;

  template <class Derived> static ReturnType create(CtorParamTypes &&...Args) {
    return std::make_unique<Derived>(std::forward<CtorParamTypes>(Args)...);
  }
};

template <typename Base, typename... CtorParamTypes> struct InstantiateOnStack {
  static_assert(!std::is_polymorphic_v<Base>);
  using ReturnType = Base;

  template <class Derived> static ReturnType create(CtorParamTypes &&...Args) {
    static_assert(sizeof(Base) == sizeof(Derived));
    return Derived(std::forward<CtorParamTypes>(Args)...);
  }
};

/// Headers:
///  - Header<Name>
///  - Header<Description>
///  - Header<Name, Description>
///  - Header<YourCustom>
/// CreationStrategy:
///  - InstantiateOnStack (returns T)
///  - InstantiateOnHeap (returns unique_ptr<T>)
///  - YourCustom (returns whatever you want for T)
/// Base: The base type of
template <class Headers, template <typename...> class CreationStrategy,
          typename Base, typename... CtorParamTypes>
class Registry {
public:
  static auto entries() { return llvm::iterator_range{Nodes}; }
  template <typename Derived> struct Add {
    template <class... Ts> Add(Ts &&...Args) {
      Nodes.emplace_back(&SpecificCreationStrategy::template create<Derived>,
                         std::forward<Ts>(Args)...);
    }
  };

private:
  Registry() = delete;
  using SpecificCreationStrategy = CreationStrategy<Base, CtorParamTypes...>;
  using ReturnType = typename SpecificCreationStrategy::ReturnType;

  class Factory {
    using FactoryFnRef = llvm::function_ref<ReturnType(CtorParamTypes...)>;

  public:
    explicit Factory(FactoryFnRef F) : F(F) {}
    ReturnType instantiate(CtorParamTypes... Args) const {
      return F(std::forward<CtorParamTypes>(Args)...);
    }

  private:
    const FactoryFnRef F;
  };

  struct RecipeData : Factory, Headers {
    template <class First, class... Ts>
    RecipeData(First &&Arg, Ts &&...Args)
        : Factory{std::forward<First>(Arg)},
          Headers{std::forward<Ts>(Args)...} {}
  };

  static inline std::list<RecipeData> Nodes;
};

/// Mimics the llvm::Registry current behavior:
/// Creates nodes holding the Name and Description.
/// They also instantiate on the heap using unique_ptrs.
template <typename Base, typename... CtorParamTypes>
using BasicRegistry =
    v2::Registry<v2::Header<v2::Name, v2::Description>, v2::InstantiateOnHeap,
                 Base, CtorParamTypes...>;

} // namespace clang::ssaf::v2

//------------------
struct PluginBase {
  virtual ~PluginBase() = 0;

  struct Data {
    int Version;
  };
  virtual Data getData() const = 0;
};

struct FancyPlugin final : PluginBase {
  Data getData() const override { return Data{404}; }
};

struct MyFormatInfo {
  using ser = llvm::function_ref<void(int)>;
  using deser = llvm::function_ref<int(std::string)>;
  using namer = llvm::function_ref<llvm::StringLiteral()>;
  MyFormatInfo(namer F, ser s, deser d) : For(F()), s(s), d(d) {}

  llvm::StringLiteral For;
  ser s;
  deser d;
};

extern void serializeFancy(int);
extern int deserializeFancy(std::string);

// This compile-time binds all fields of the Base class.
template <auto... Args> struct MyFormatInfoFor : MyFormatInfo {
  MyFormatInfoFor() : MyFormatInfo{Args...} {}
};

// We can't pass a string-literal as a NTTP, but we can pass a function
// returning the desired literal. Classic metaprogramming trick.
static constexpr llvm::StringLiteral FancyName() { return "Fancy"; }
struct FancyFormatInfo
    : MyFormatInfoFor<FancyName, serializeFancy, deserializeFancy> {};

//------------------

using MyBasicRegistry = v2::BasicRegistry<PluginBase>;
static auto RegisterCommonEntry =
    MyBasicRegistry::Add<FancyPlugin>("Name", "Desc");
static void test_basicregistry() {
  for (const auto &Recipe : MyBasicRegistry::entries()) {
    llvm::errs() << "Name: " << Recipe.Name << "\n";
    llvm::errs() << "Desc: " << Recipe.Description << "\n";
    std::unique_ptr<PluginBase> Uptr = Recipe.instantiate();
    int V = Uptr->getData().Version;
  }
}

//------------------

using MyFormatInfoRegistry =
    v2::Registry<v2::Header<v2::Name>, v2::InstantiateOnStack, MyFormatInfo>;
static auto RegisterFancy =
    MyFormatInfoRegistry::Add<FancyFormatInfo>("FancyFormatInfo");
static void test_myformatinfo() {
  for (const auto &Recipe : MyFormatInfoRegistry::entries()) {
    llvm::StringRef Name = Recipe.Name;
    MyFormatInfo Instance = Recipe.instantiate();
    llvm::StringLiteral For = Instance.For;
    llvm::function_ref<void(int)> s = Instance.s;
    llvm::function_ref<int(std::string)> d = Instance.d;
  }
}

//------------------

using PluginRegistry = v2::Registry<v2::Header<v2::Name, v2::Description>,
                                    v2::InstantiateOnHeap, PluginBase>;
static auto RegisterFancyPlugin =
    PluginRegistry::Add<FancyPlugin>("FancyPlugin", "desc");

static void test_fancy_plugin() {
  for (const auto &Recipe : PluginRegistry::entries()) {
    llvm::StringRef Name = Recipe.Name;
    llvm::StringRef Desc = Recipe.Description;
    std::unique_ptr<PluginBase> Instance = Recipe.instantiate();
    int V = Instance->getData().Version;
  }
}

//------------------

struct FailableBase {
  virtual ~FailableBase() = 0;
  virtual float brum_brum() = 0;

  using ReturnType = llvm::Expected<std::unique_ptr<FailableBase>>;

  // Custom create function that returns llvm::Expected.
  // This essentially behaves as the 'constructor' of 'Derived' that can do
  // precondition validation and fail for different reasons.
  template <class Derived> static ReturnType create(int Num) {
    if (Num > 100)
      return llvm::createStringError("Greater than 100");

    // It doesn't need to actually forward the 'int' if we don't want to.
    return std::make_unique<Derived>();
  }
};

struct FailableDerived final : FailableBase {
  float brum_brum() override { return 3.1f; }
};

template <typename Base, typename... CtorParamTypes>
struct InstantiateUsingStaticMemberCreate {
  using ReturnType = typename Base::ReturnType;

  template <class Derived> static ReturnType create(CtorParamTypes &&...Args) {
    return Base::template create<Derived>(
        std::forward<CtorParamTypes>(Args)...);
  }
};

using FailableRegistry =
    v2::Registry<v2::Header<v2::Name, v2::Description>,
                 InstantiateUsingStaticMemberCreate, FailableBase, int>;
static auto RegisterFailable =
    FailableRegistry::Add<FailableDerived>("FancyPlugin", "desc");
static void test_failable(int N) {
  for (const auto &Recipe : FailableRegistry::entries()) {
    llvm::StringRef Name = Recipe.Name;
    llvm::StringRef Desc = Recipe.Description;
    llvm::Expected<std::unique_ptr<FailableBase>> ExpectedInstance =
        Recipe.instantiate(N);
    if (!ExpectedInstance) {
      llvm::logAllUnhandledErrors(ExpectedInstance.takeError(), llvm::errs());
      continue;
    }
    float f = (*ExpectedInstance)->brum_brum();
  }
}

//------------------

struct ConsumedID {
  const unsigned ID;

  ConsumedID(unsigned ID) : ID(ID) {
    llvm::errs() << "ConsumedID ctor side-effect ran with consumed ID " << ID
                 << "\n";
  }
};

template <unsigned I> struct ConsumedIdOf : ConsumedID {
  ConsumedIdOf() : ConsumedID{I} {}
};
using ConsumedIDRegistry =
    v2::Registry<v2::Header<>, v2::InstantiateOnStack, ConsumedID>;
static auto ReserveID = ConsumedIDRegistry::Add<ConsumedIdOf<10>>();

static void test_consumed_ids() {
  for (const auto &Recipe : ConsumedIDRegistry::entries()) {
    ConsumedID Obj = Recipe.instantiate();
    unsigned UsedID = Obj.ID;
  }
}

//------------------
