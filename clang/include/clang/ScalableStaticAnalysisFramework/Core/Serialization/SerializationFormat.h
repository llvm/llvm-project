//===- SerializationFormat.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract SerializationFormat interface for reading and writing
// TUSummary and LinkUnitResolution data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SERIALIZATION_SERIALIZATIONFORMAT_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SERIALIZATION_SERIALIZATIONFORMAT_H

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Registry.h"

namespace clang::ssaf {

/// Abstract base class for serialization formats.
class SerializationFormat {
public:
  virtual ~SerializationFormat() = default;

  virtual llvm::Expected<TUSummary> readTUSummary(llvm::StringRef Path) = 0;

  virtual llvm::Error writeTUSummary(const TUSummary &Summary,
                                     llvm::StringRef Path) = 0;

  virtual llvm::Expected<TUSummaryEncoding>
  readTUSummaryEncoding(llvm::StringRef Path) = 0;

  virtual llvm::Error
  writeTUSummaryEncoding(const TUSummaryEncoding &SummaryEncoding,
                         llvm::StringRef Path) = 0;

  virtual llvm::Expected<LUSummary> readLUSummary(llvm::StringRef Path) = 0;

  virtual llvm::Error writeLUSummary(const LUSummary &Summary,
                                     llvm::StringRef Path) = 0;

  virtual llvm::Expected<LUSummaryEncoding>
  readLUSummaryEncoding(llvm::StringRef Path) = 0;

  virtual llvm::Error
  writeLUSummaryEncoding(const LUSummaryEncoding &SummaryEncoding,
                         llvm::StringRef Path) = 0;

  virtual llvm::Expected<WPASuite> readWPASuite(llvm::StringRef Path) = 0;

  virtual llvm::Error writeWPASuite(const WPASuite &Suite,
                                    llvm::StringRef Path) = 0;

  /// Invokes \p Callback once for each analysis that has registered
  /// serialization support for this format.
  virtual void forEachRegisteredAnalysis(
      llvm::function_ref<void(llvm::StringRef Name, llvm::StringRef Desc)>
          Callback) const = 0;

protected:
  // Helpers providing access to implementation details of basic data structures
  // for efficient serialization/deserialization.

  static EntityId makeEntityId(const size_t Index) { return EntityId(Index); }

  /// Constructs an empty WPASuite. Bypasses the private default constructor
  /// so that deserialization code can build a WPASuite incrementally.
  static WPASuite makeWPASuite() { return WPASuite(); }

#define FIELD(CLASS, FIELD_NAME)                                               \
  static const auto &get##FIELD_NAME(const CLASS &X) { return X.FIELD_NAME; }  \
  static auto &get##FIELD_NAME(CLASS &X) { return X.FIELD_NAME; }
#include "clang/ScalableStaticAnalysisFramework/Core/Model/PrivateFieldNames.def"

  /// Generates a per-format plugin registry for analysis result
  /// serializers and deserializers.
  ///
  /// Each concrete format (e.g. JSONFormat) instantiates this template once
  /// via a \c using alias, then exposes that alias publicly so that analysis
  /// authors can register (de)serialization support with a single declaration:
  ///
  ///   static MyFormat::AnalysisResultRegistryGenerator::Add<MyAnalysisResult>
  ///       Reg(serializeFn, deserializeFn);
  ///
  /// ---
  /// Design overview
  /// ---
  ///
  /// **Registry isolation via \p FormatT.**
  /// The underlying store is \c llvm::Registry<Entry>, which is a global
  /// linked list keyed on the \c Entry type. Because \c Entry is a member of
  /// this template, each \c (FormatT, SerializerFn, DeserializerFn)
  /// instantiation produces a distinct \c Entry type and therefore a distinct
  /// \c llvm::Registry — even if two formats happen to share the same
  /// serializer/deserializer function signatures. The \p FormatT parameter
  /// exists solely to provide this isolation; it is otherwise unused inside
  /// the template body.
  ///
  /// **Bridging \c function_ref into \c llvm::Registry.**
  /// \c llvm::function_ref is a non-owning view of a callable — it cannot be
  /// stored inside the registry because the registry only keeps nullary
  /// factories of the form <tt>[]{ return make_unique<ConcreteEntry>(); }</tt>
  /// that capture no state. Two mechanisms bridge this gap:
  ///
  ///   1. *Function-local statics as per-analysis storage.*
  ///      Inside \c Add<AnalysisResultT>::Add(...), two function-local statics
  ///      — \c SavedSerialize and \c SavedDeserialize — are initialized from
  ///      the constructor arguments on the first (and only) call. Because
  ///      \c Add<T>::Add(...) is a distinct function for each \c T, each
  ///      analysis type gets its own pair of statics with program lifetime,
  ///      giving the \c function_ref values a stable home.
  ///
  ///   2. *\c ConcreteEntry as a local struct.*
  ///      \c ConcreteEntry is defined inside the \c Add constructor body so
  ///      that its default constructor can read \c SavedSerialize and
  ///      \c SavedDeserialize from the enclosing function scope. When
  ///      \c llvm::Registry later calls \c E.instantiate() during \c lookup,
  ///      it invokes \c ConcreteEntry(), which re-wraps those stored values
  ///      into a fresh \c Entry — reconstructing the \c function_ref from the
  ///      same stable underlying callables.
  ///
  /// **One-time registration via a function-local static \c Reg.**
  /// \c static typename RegistryT::template Add<ConcreteEntry> Reg(NameStr,"")
  /// is also a function-local static. C++ guarantees it is initialized exactly
  /// once, on the first call to \c Add<T>::Add(...). Its constructor appends a
  /// node to the \c llvm::Registry linked list, associating \c NameStr with
  /// the \c ConcreteEntry factory. All subsequent calls to \c Add<T>::Add(...)
  /// hit the \c Registered guard first and abort with a fatal error, making
  /// duplicate registrations a detectable programmer mistake rather than a
  /// silent no-op.
  ///
  /// **Lookup.**
  /// \c lookup iterates \c RegistryT::entries() and compares each node's name
  /// (available directly on the entry node without instantiation) against the
  /// requested \c AnalysisName. Only the matching node is instantiated via
  /// \c E.instantiate(), which invokes \c ConcreteEntry() and returns the
  /// stored \c function_ref pair.
  template <class FormatT, class SerializerFn, class DeserializerFn>
  class AnalysisResultRegistryGenerator {
  public:
    struct Entry {
      explicit Entry(SerializerFn Serialize, DeserializerFn Deserialize)
          : Serialize(Serialize), Deserialize(Deserialize) {}
      virtual ~Entry() = default;
      SerializerFn Serialize;
      DeserializerFn Deserialize;
    };

    using RegistryT = llvm::Registry<Entry>;

    template <class AnalysisResultT> struct Add {
      Add(SerializerFn Serialize, DeserializerFn Deserialize) {
        static bool Registered = false;
        if (Registered) {
          ErrorBuilder::fatal("support is already registered for analysis: {0}",
                              AnalysisResultT::analysisName());
        }
        Registered = true;
        static SerializerFn SavedSerialize = Serialize;
        static DeserializerFn SavedDeserialize = Deserialize;

        struct ConcreteEntry : Entry {
          ConcreteEntry() : Entry(SavedSerialize, SavedDeserialize) {}
        };

        static std::string NameStr =
            AnalysisResultT::analysisName().str().str();
        static typename RegistryT::template Add<ConcreteEntry> Reg(NameStr, "");
      }
    };

    static llvm::Expected<std::pair<SerializerFn, DeserializerFn>>
    lookup(const AnalysisName &Name) {
      for (const auto &E : RegistryT::entries()) {
        if (E.getName() == Name.str()) {
          auto Entry = E.instantiate();
          return std::make_pair(Entry->Serialize, Entry->Deserialize);
        }
      }
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  "no support registered for analysis: {0}",
                                  Name)
          .build();
    }
  };
};

template <class SerializerFn, class DeserializerFn> struct FormatInfoEntry {
  FormatInfoEntry(SummaryName ForSummary, SerializerFn Serialize,
                  DeserializerFn Deserialize)
      : ForSummary(ForSummary), Serialize(Serialize), Deserialize(Deserialize) {
  }
  virtual ~FormatInfoEntry() = default;

  SummaryName ForSummary;
  SerializerFn Serialize;
  DeserializerFn Deserialize;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SERIALIZATION_SERIALIZATIONFORMAT_H
