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

  /// Per-format plugin registry for analysis result (de)serializers.
  ///
  /// Each concrete format (e.g. JSONFormat) instantiates this template once
  /// via a public \c using alias. Analysis authors register support with:
  ///
  ///   static MyFormat::AnalysisResultRegistry::Add<MyAnalysisResult>
  ///       Reg(serializeFn, deserializeFn);
  ///
  /// The serializer receives \c const MyAnalysisResult & directly — the
  /// \c Add wrapper handles the downcast from \c AnalysisResult internally.
  ///
  /// \p FormatT is otherwise unused — it exists because \c llvm::Registry
  /// is keyed on the \c Entry type, so two formats that happen to share the
  /// same serializer/deserializer signatures would collide without a
  /// disambiguating template parameter.
  ///
  /// \c function_ref is non-owning, but \c llvm::Registry only stores
  /// nullary factories (no captured state). Function-local statics inside
  /// \c Add<T>::Add(...) give each analysis's \c function_ref values a
  /// stable, program-lifetime home, and a local \c ConcreteEntry struct
  /// reads them back when the registry instantiates the factory.
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
      /// Extracts the typed serializer signature from \c SerializerFn.
      /// Given \c function_ref<R(const AnalysisResult &, Args...)>, produces
      /// a function-pointer type \c R(*)(const AnalysisResultT &, Args...) and
      /// a static \c wrap() that downcasts and forwards.
      template <class> struct SerializerAdapter;
      template <class R, class... Args>
      struct SerializerAdapter<
          llvm::function_ref<R(const AnalysisResult &, Args...)>> {
        using TypedFnPtr = R (*)(const AnalysisResultT &, Args...);
        static inline TypedFnPtr Saved = nullptr;
        static R wrap(const AnalysisResult &Base, Args... args) {
          return Saved(static_cast<const AnalysisResultT &>(Base), args...);
        }
      };
      using SA = SerializerAdapter<SerializerFn>;

      Add(typename SA::TypedFnPtr TypedSerialize, DeserializerFn Deserialize) {
        static bool Registered = false;
        if (Registered) {
          ErrorBuilder::fatal("support is already registered for analysis: {0}",
                              AnalysisResultT::analysisName());
        }
        Registered = true;
        SA::Saved = TypedSerialize;
        static auto *SerializeWrap = &SA::wrap;
        static SerializerFn SavedSerialize(SerializeWrap);
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
