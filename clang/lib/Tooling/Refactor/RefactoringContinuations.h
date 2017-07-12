//===--- RefactoringContinuations.h - Defines refactoring continuations ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_TOOLING_REFACTOR_REFACTORING_CONTINUATIONS_H
#define LLVM_CLANG_LIB_TOOLING_REFACTOR_REFACTORING_CONTINUATIONS_H

#include "clang/AST/Decl.h"
#include "clang/Tooling/Refactor/IndexerQuery.h"
#include "clang/Tooling/Refactor/RefactoringOperation.h"
#include "llvm/ADT/StringMap.h"
#include <tuple>

namespace clang {
namespace tooling {

namespace detail {

struct ValidBase {};

/// The ContinuationPassType determine which type is passed into the refactoring
/// continuation.
template <typename T> struct ContinuationPassType { using Type = T; };

template <typename T> struct ContinuationPassType<std::vector<T>> {
  using Type = ArrayRef<T>;
};

/// Refactoring operations can pass state to the continuations. Valid state
/// values should have a corresponding \c StateTraits specialization.
template <typename T> struct StateTraits {
  /// Specializations should define the following types:
  ///
  /// StoredResultType: The TU-specific type which is then passed into the
  /// continuation function. The continuation receives the result whose type is
  /// \c ContinuationPassType<StoredResultType>::Type.
  ///
  /// PersistentType: The TU-independent type that's persisted even after the
  /// TU in which the continuation was created is disposed.
};

template <typename T>
struct StateTraits<const T *>
    : std::enable_if<std::is_base_of<Decl, T>::value, ValidBase>::type {
  using StoredResultType = const T *;
  using PersistentType = PersistentDeclRef<T>;
};

template <typename T>
struct StateTraits<ArrayRef<const T *>>
    : std::enable_if<std::is_base_of<Decl, T>::value, ValidBase>::type {
  using StoredResultType = std::vector<const T *>;
  using PersistentType = std::vector<PersistentDeclRef<T>>;
};

template <typename T>
struct StateTraits<std::unique_ptr<indexer::ManyToManyDeclarationsQuery<T>>>
    : std::enable_if<std::is_base_of<Decl, T>::value, ValidBase>::type {
  using StoredResultType = std::vector<indexer::Indexed<const T *>>;
  using PersistentType = std::vector<indexer::Indexed<PersistentDeclRef<T>>>;
};

template <> struct StateTraits<std::vector<std::string>> {
  using StoredResultType = std::vector<std::string>;
  using PersistentType = std::vector<std::string>;
};

/// Conversion functions convert the TU-specific state to a TU independent
/// state and vice-versa.
template <typename T>
PersistentDeclRef<T> convertToPersistentRepresentation(
    const T *Declaration,
    typename std::enable_if<std::is_base_of<Decl, T>::value>::type * =
        nullptr) {
  return PersistentDeclRef<T>::create(Declaration);
}

template <typename T>
std::vector<PersistentDeclRef<T>> convertToPersistentRepresentation(
    ArrayRef<const T *> Declarations,
    typename std::enable_if<std::is_base_of<Decl, T>::value>::type * =
        nullptr) {
  std::vector<PersistentDeclRef<T>> Result;
  Result.reserve(Declarations.size());
  for (const T *D : Declarations)
    Result.push_back(PersistentDeclRef<T>::create(D));
  return Result;
}

template <typename T>
std::vector<indexer::Indexed<PersistentDeclRef<T>>>
convertToPersistentRepresentation(
    std::unique_ptr<indexer::ManyToManyDeclarationsQuery<T>> &Query,
    typename std::enable_if<std::is_base_of<Decl, T>::value>::type * =
        nullptr) {
  Query->invalidateTUSpecificState();
  return Query->getOutput();
}

inline std::vector<std::string>
convertToPersistentRepresentation(const std::vector<std::string> &Values) {
  return Values;
}

/// Converts the TU-independent state to the TU-specific state.
class PersistentToASTSpecificStateConverter {
  ASTContext &Context;
  llvm::StringMap<const Decl *> ConvertedDeclRefs;

  const Decl *lookupDecl(StringRef USR);

public:
  // FIXME: We can hide the addConvertible/convert interface so that
  // the continuation will just invoke one conversion function for the entire
  // tuple.
  PersistentToASTSpecificStateConverter(ASTContext &Context)
      : Context(Context) {}

  template <typename T>
  bool addConvertible(
      const PersistentDeclRef<T> &Ref,
      typename std::enable_if<std::is_base_of<Decl, T>::value>::type * =
          nullptr) {
    if (!Ref.USR.empty())
      ConvertedDeclRefs[Ref.USR] = nullptr;
    return true;
  }

  template <typename T>
  const T *
  convert(const PersistentDeclRef<T> &Ref,
          typename std::enable_if<std::is_base_of<Decl, T>::value>::type * =
              nullptr) {
    return dyn_cast_or_null<T>(lookupDecl(Ref.USR));
  }

  template <typename T>
  bool addConvertible(
      const std::vector<PersistentDeclRef<T>> &Refs,
      typename std::enable_if<std::is_base_of<Decl, T>::value>::type * =
          nullptr) {
    for (const auto &Ref : Refs) {
      if (!Ref.USR.empty())
        ConvertedDeclRefs[Ref.USR] = nullptr;
    }
    return true;
  }

  template <typename T>
  std::vector<const T *>
  convert(const std::vector<PersistentDeclRef<T>> &Refs,
          typename std::enable_if<std::is_base_of<Decl, T>::value>::type * =
              nullptr) {
    std::vector<const T *> Results;
    Results.reserve(Refs.size());
    // Allow nulls in the produced array, the continuation will have to deal
    // with them by itself.
    for (const auto &Ref : Refs)
      Results.push_back(dyn_cast_or_null<T>(lookupDecl(Ref.USR)));
    return Results;
  }

  template <typename T>
  bool addConvertible(
      const std::vector<indexer::Indexed<PersistentDeclRef<T>>> &Refs,
      typename std::enable_if<std::is_base_of<Decl, T>::value>::type * =
          nullptr) {
    for (const auto &Ref : Refs) {
      if (!Ref.Decl.USR.empty())
        ConvertedDeclRefs[Ref.Decl.USR] = nullptr;
    }
    return true;
  }

  template <typename T>
  std::vector<indexer::Indexed<const T *>>
  convert(const std::vector<indexer::Indexed<PersistentDeclRef<T>>> &Refs,
          typename std::enable_if<std::is_base_of<Decl, T>::value>::type * =
              nullptr) {
    std::vector<indexer::Indexed<const T *>> Results;
    Results.reserve(Refs.size());
    // Allow nulls in the produced array, the continuation will have to deal
    // with them by itself.
    for (const auto &Ref : Refs)
      Results.push_back(indexer::Indexed<const T *>(
          dyn_cast_or_null<T>(lookupDecl(Ref.Decl.USR)), Ref.IsNotDefined));
    return Results;
  }

  bool addConvertible(const PersistentFileID &) {
    // Do nothing since FileIDs are converted one-by-one.
    return true;
  }

  FileID convert(const PersistentFileID &Ref);

  bool addConvertible(const std::vector<std::string> &) { return true; }

  std::vector<std::string> convert(const std::vector<std::string> &Values) {
    return Values;
  }

  /// Converts the added persistent state into TU-specific state using one
  /// efficient operation.
  void runCoalescedConversions();
};

template <typename T, typename ASTQueryType, typename... QueryOrState>
struct ContinuationFunction {
  using Type = llvm::Expected<RefactoringResult> (*)(
      ASTContext &, const T &,
      typename ContinuationPassType<
          typename StateTraits<QueryOrState>::StoredResultType>::Type...);

  template <size_t... Is>
  static llvm::Expected<RefactoringResult> dispatch(
      Type Fn, detail::PersistentToASTSpecificStateConverter &Converter,
      ASTContext &Context, const ASTQueryType &Query,
      const std::tuple<typename StateTraits<QueryOrState>::StoredResultType...>
          &Arguments,
      llvm::index_sequence<Is...>) {
    auto ASTQueryResult = Converter.convert(Query.getResult());
    return Fn(Context, ASTQueryResult, std::get<Is>(Arguments)...);
  }
};

template <typename ASTQueryType, typename... QueryOrState>
struct ContinuationFunction<void, ASTQueryType, QueryOrState...> {
  using Type = llvm::Expected<RefactoringResult> (*)(
      ASTContext &,
      typename ContinuationPassType<
          typename StateTraits<QueryOrState>::StoredResultType>::Type...);

  template <size_t... Is>
  static llvm::Expected<RefactoringResult> dispatch(
      Type Fn, detail::PersistentToASTSpecificStateConverter &,
      ASTContext &Context, const ASTQueryType &,
      const std::tuple<typename StateTraits<QueryOrState>::StoredResultType...>
          &Arguments,
      llvm::index_sequence<Is...>) {
    return Fn(Context, std::get<Is>(Arguments)...);
  }
};

/// The refactoring contination contains a set of structures that implement
/// the refactoring operation continuation mechanism.
template <typename ASTQueryType, typename... QueryOrState>
class SpecificRefactoringContinuation final : public RefactoringContinuation {
public:
  static_assert(std::is_base_of<indexer::ASTProducerQuery, ASTQueryType>::value,
                "Invalid AST Query");
  // TODO: Validate the QueryOrState types.

  /// The consumer function is the actual continuation. It receives the state
  /// that was passed-in in the request or the results of the indexing queries
  /// that were passed-in in the request.
  using ConsumerFn =
      typename ContinuationFunction<typename ASTQueryType::ResultTy,
                                    ASTQueryType, QueryOrState...>::Type;

private:
  ConsumerFn Consumer;
  std::unique_ptr<ASTQueryType> ASTQuery;
  /// Inputs store state that's dependent on the original TU.
  llvm::Optional<std::tuple<QueryOrState...>> Inputs;
  /// State contains TU-independent values.
  llvm::Optional<
      std::tuple<typename StateTraits<QueryOrState>::PersistentType...>>
      State;

  /// Converts a tuple that contains the TU dependent state to a tuple with
  /// TU independent state.
  template <size_t... Is>
  std::tuple<typename StateTraits<QueryOrState>::PersistentType...>
  convertToPersistentImpl(llvm::index_sequence<Is...>) {
    assert(Inputs && "TU-dependent state is already converted");
    return std::make_tuple(
        detail::convertToPersistentRepresentation(std::get<Is>(*Inputs))...);
  }

  template <typename T>
  bool gatherQueries(
      std::vector<indexer::IndexerQuery *> &Queries,
      const std::unique_ptr<T> &Query,
      typename std::enable_if<
          std::is_base_of<indexer::IndexerQuery, T>::value>::type * = nullptr) {
    Queries.push_back(Query.get());
    return true;
  }

  template <typename T>
  bool gatherQueries(std::vector<indexer::IndexerQuery *> &, const T &) {
    // This input element is not a query.
    return true;
  }

  template <size_t... Is>
  std::vector<indexer::IndexerQuery *>
  gatherQueries(llvm::index_sequence<Is...>) {
    assert(Inputs && "TU-dependent state is already converted");
    std::vector<indexer::IndexerQuery *> Queries;
    std::make_tuple(gatherQueries(Queries, std::get<Is>(*Inputs))...);
    return Queries;
  }

  /// Calls the consumer function with the given \p Context and the state
  /// whose values are converted from the TU-independent to TU-specific values.
  template <size_t... Is>
  llvm::Expected<RefactoringResult> dispatch(ASTContext &Context,
                                             llvm::index_sequence<Is...> Seq) {
    assert(State && "TU-independent state is not yet produced");
    detail::PersistentToASTSpecificStateConverter Converter(Context);
    (void)std::make_tuple(Converter.addConvertible(std::get<Is>(*State))...);
    Converter.runCoalescedConversions();
    auto Results = std::make_tuple(Converter.convert(std::get<Is>(*State))...);
    // TODO: Check for errors?
    return detail::ContinuationFunction<
        typename ASTQueryType::ResultTy, ASTQueryType,
        QueryOrState...>::dispatch(Consumer, Converter, Context, *ASTQuery,
                                   Results, Seq);
  }

public:
  SpecificRefactoringContinuation(ConsumerFn Consumer,
                                  std::unique_ptr<ASTQueryType> ASTQuery,
                                  QueryOrState... Inputs)
      : Consumer(Consumer), ASTQuery(std::move(ASTQuery)),
        Inputs(std::make_tuple(std::move(Inputs)...)) {}

  SpecificRefactoringContinuation(SpecificRefactoringContinuation &&) = default;
  SpecificRefactoringContinuation &
  operator=(SpecificRefactoringContinuation &&) = default;

  indexer::ASTProducerQuery *getASTUnitIndexerQuery() override {
    return ASTQuery.get();
  }

  std::vector<indexer::IndexerQuery *> getAdditionalIndexerQueries() override {
    return gatherQueries(llvm::index_sequence_for<QueryOrState...>());
  }

  /// Query results are fetched. State is converted to a persistent
  /// representation.
  void persistTUSpecificState() override {
    ASTQuery->invalidateTUSpecificState();
    State =
        convertToPersistentImpl(llvm::index_sequence_for<QueryOrState...>());
    Inputs = None;
  }

  /// The state is converted to the AST representation in the given ASTContext
  /// and the continuation is dispatched.
  llvm::Expected<RefactoringResult>
  runInExternalASTUnit(ASTContext &Context) override {
    return dispatch(Context, llvm::index_sequence_for<QueryOrState...>());
  }
};

} // end namespace detail

/// Returns a refactoring continuation that will run within the context of a
/// single external AST unit.
///
/// The indexer determines which AST unit should receive the continuation by
/// evaluation the AST query operation \p ASTQuery.
///
/// \param ASTQuery The query that will determine which AST unit should the
/// continuation run in.
///
/// \param Consumer The continuation function that will be called once the
/// external AST unit is loaded.
///
/// \param Inputs Each individiual input element can contain either some
/// state value that will be passed into the \p Consumer function or an
/// indexer query whose results will be passed into the \p Consumer function.
template <typename ASTQueryType, typename... QueryOrState>
typename std::enable_if<
    std::is_base_of<indexer::ASTProducerQuery, ASTQueryType>::value,
    std::unique_ptr<RefactoringContinuation>>::type
continueInExternalASTUnit(
    std::unique_ptr<ASTQueryType> ASTQuery,
    typename detail::SpecificRefactoringContinuation<
        ASTQueryType, QueryOrState...>::ConsumerFn Consumer,
    QueryOrState... Inputs) {
  return llvm::make_unique<
      detail::SpecificRefactoringContinuation<ASTQueryType, QueryOrState...>>(
      Consumer, std::move(ASTQuery), std::move(Inputs)...);
}

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_LIB_TOOLING_REFACTOR_REFACTORING_CONTINUATIONS_H
