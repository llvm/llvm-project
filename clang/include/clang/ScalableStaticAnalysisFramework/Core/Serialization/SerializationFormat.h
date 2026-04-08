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
  /// \code
  ///   static MyFormat::AnalysisResultRegistry::Add<MyAnalysisResult>
  ///       Reg(serializeFn, deserializeFn);
  /// \endcode
  ///
  /// The serializer receives a \c const reference to \c MyAnalysisResult
  /// directly and the \c Add wrapper handles the downcast from \c
  /// AnalysisResult internally via virtual dispatch.
  ///
  /// \tparam FormatT Phantom type is needed to disambiguate \c llvm::Registry
  ///   instantiations. \c llvm::Registry is keyed on the \c Entry type,
  ///   so two formats sharing the same serializer/deserializer signatures
  ///   would collide on the same registry without this parameter.
  template <class FormatT, class SerializerFn, class DeserializerFn>
  class AnalysisResultRegistryGenerator;

  template <class FormatT, class SerRet, class... SerArgs, class DesRet,
            class... DesArgs>
  class AnalysisResultRegistryGenerator<
      FormatT, llvm::function_ref<SerRet(const AnalysisResult &, SerArgs...)>,
      llvm::function_ref<DesRet(DesArgs...)>> {

    using DeserializerFn = llvm::function_ref<DesRet(DesArgs...)>;

  public:
    /// Abstract base type stored in \c llvm::Registry<Codec>.
    /// Subclasses override \c serialize() and \c deserialize() to
    /// dispatch to the plugin's concrete functions.
    ///
    /// There is one \c Codec type (and one \c llvm::Registry<Codec>) per
    /// format. All analysis-specific \c ConcreteCodec<AnalysisResultT>
    /// subclasses for a given format register into that single registry. The \c
    /// FormatT phantom type parameter on the enclosing class ensures that
    /// different formats produce distinct \c Codec types and thus separate
    /// registries.
    struct Codec {
      virtual ~Codec() = default;
      virtual SerRet serialize(const AnalysisResult &, SerArgs...) const = 0;
      virtual DesRet deserialize(DesArgs...) const = 0;
    };

    /// Per-\c AnalysisResultT subclass of \c Codec. The \c serialize()
    /// override performs the downcast from \c AnalysisResult to
    /// \c AnalysisResultT.
    template <class AnalysisResultT> struct ConcreteCodec final : Codec {
      using TypedSerializerFn =
          llvm::function_ref<SerRet(const AnalysisResultT &, SerArgs...)>;

      /// The plugin's serializer and deserializer are stored as \c static
      /// \c inline members because \c llvm::Registry instantiates entries
      /// via a zero-argument constructor so there is no way to pass them
      /// as constructor arguments. The \c Add constructor writes these
      /// members before registration, and the virtual methods read them
      /// on invocation. \c inline avoids the need for separate out-of-line
      /// definitions that a bare \c static member in a class template
      /// would require.
      static inline TypedSerializerFn SavedSerialize;
      static inline DeserializerFn SavedDeserialize;

      SerRet serialize(const AnalysisResult &Base,
                       SerArgs... args) const override {
        return SavedSerialize(static_cast<const AnalysisResultT &>(Base),
                              args...);
      }

      DesRet deserialize(DesArgs... args) const override {
        return SavedDeserialize(args...);
      }
    };

    template <class AnalysisResultT> struct Add {
      /// Takes the plugin's typed serializer and the deserializer, and
      /// inserts them into \c llvm::Registry<Codec>.
      Add(typename ConcreteCodec<AnalysisResultT>::TypedSerializerFn
              TypedSerialize,
          DeserializerFn Deserialize) {
        /// Per-\c AnalysisResultT guard: each template instantiation gets
        /// its own \c static \c bool, so double-registration of the same
        /// analysis is caught even across translation units.
        static bool Registered = false;
        if (Registered) {
          ErrorBuilder::fatal("support is already registered for analysis: {0}",
                              AnalysisResultT::analysisName());
        }
        Registered = true;

        ConcreteCodec<AnalysisResultT>::SavedSerialize = TypedSerialize;
        ConcreteCodec<AnalysisResultT>::SavedDeserialize = Deserialize;

        /// \c llvm::Registry stores the name as a \c StringRef, so the
        /// underlying string must be kept alive with a static declaration.
        static std::string NameStr =
            AnalysisResultT::analysisName().str().str();

        /// This performs the actual registration. It appends a factory for \c
        /// ConcreteCodec to the global \c llvm::Registry<Codec>. \c static
        /// ensures the `Registry::Add` object lives for the entire program,
        /// keeping its codec and node alive in the registry's linked list.
        [[maybe_unused]] static typename llvm::Registry<Codec>::template Add<
            ConcreteCodec<AnalysisResultT>> RegisterUsingCtorSideEffect(NameStr,
                                                                        "");
      }
    };

    /// Looks up the codec for \p Name by walking the registry list.
    static llvm::Expected<std::unique_ptr<Codec>>
    instantiate(const AnalysisName &Name) {
      for (const auto &E : llvm::Registry<Codec>::entries()) {
        if (E.getName() == Name.str()) {
          return E.instantiate();
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
