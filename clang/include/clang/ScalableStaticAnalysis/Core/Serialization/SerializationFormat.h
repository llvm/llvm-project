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

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_SERIALIZATION_SERIALIZATIONFORMAT_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_SERIALIZATION_SERIALIZATIONFORMAT_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/MultiArchSharedLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/MultiArchStaticLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/StaticLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysis/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Registry.h"

#include <variant>

namespace clang::ssaf {

/// Sum type returned by \c SerializationFormat::readArtifact, used when the
/// caller does not know up-front which kind of top-level SSAF artifact a
/// file contains. The active alternative is decided by the file's
/// self-describing type field.
using Artifact = std::variant<TUSummary, LUSummary, WPASuite>;

/// Lazily-deserialized counterpart of \c Artifact: the same on-disk
/// artifacts but with their per-entity summary payloads left as opaque
/// format-specific encodings rather than fully resolved analysis results.
///
/// \c StaticLibrary, \c MultiArchStaticLibrary, and
/// \c MultiArchSharedLibrary appear only in this variant: the
/// archiver/arch tools and the linker pass member payloads through
/// without decoding them, so fully decoded shapes would have no consumer.
using ArtifactEncoding =
    std::variant<TUSummaryEncoding, LUSummaryEncoding, StaticLibrary,
                 MultiArchStaticLibrary, MultiArchSharedLibrary>;

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

  /// Generic read entry point. Inspects the file's self-describing type
  /// field and dispatches to \c readTUSummary or \c readLUSummary
  /// accordingly. Returns an error if the type field is missing or names
  /// an unrecognized artifact kind.
  virtual llvm::Expected<Artifact> readArtifact(llvm::StringRef Path) = 0;

  /// Generic write entry point. Dispatches to \c writeTUSummary or
  /// \c writeLUSummary based on the active variant alternative.
  virtual llvm::Error writeArtifact(const Artifact &A,
                                    llvm::StringRef Path) = 0;

  /// Encoding-flavored counterpart of \c readArtifact. Inspects the
  /// self-describing type field and dispatches to
  /// \c readTUSummaryEncoding or \c readLUSummaryEncoding accordingly.
  virtual llvm::Expected<ArtifactEncoding>
  readArtifactEncoding(llvm::StringRef Path) = 0;

  /// Encoding-flavored counterpart of \c writeArtifact. Dispatches to
  /// \c writeTUSummaryEncoding or \c writeLUSummaryEncoding based on the
  /// active variant alternative.
  virtual llvm::Error writeArtifactEncoding(const ArtifactEncoding &E,
                                            llvm::StringRef Path) = 0;

  virtual llvm::Expected<LUSummaryEncoding>
  readLUSummaryEncoding(llvm::StringRef Path) = 0;

  virtual llvm::Error
  writeLUSummaryEncoding(const LUSummaryEncoding &SummaryEncoding,
                         llvm::StringRef Path) = 0;

  virtual llvm::Expected<StaticLibrary>
  readStaticLibrary(llvm::StringRef Path) = 0;

  virtual llvm::Error writeStaticLibrary(const StaticLibrary &S,
                                         llvm::StringRef Path) = 0;

  virtual llvm::Expected<MultiArchStaticLibrary>
  readMultiArchStaticLibrary(llvm::StringRef Path) = 0;

  virtual llvm::Error
  writeMultiArchStaticLibrary(const MultiArchStaticLibrary &M,
                              llvm::StringRef Path) = 0;

  virtual llvm::Expected<MultiArchSharedLibrary>
  readMultiArchSharedLibrary(llvm::StringRef Path) = 0;

  virtual llvm::Error
  writeMultiArchSharedLibrary(const MultiArchSharedLibrary &M,
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
#include "clang/ScalableStaticAnalysis/Core/Model/PrivateFieldNames.def"

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

    using DeserializerFn = DesRet (*)(DesArgs...);

  public:
    /// Abstract base type stored in \c llvm::Registry<Codec>.
    /// Subclasses override \c serialize() and \c deserialize() to
    /// dispatch to the plugin's concrete functions.
    ///
    /// There is one \c Codec type (and one \c llvm::Registry<Codec>) per
    /// format. All analysis-specific concrete subclasses for a given format
    /// register into that single registry. The \c FormatT phantom type
    /// parameter on the enclosing class ensures that different formats
    /// produce distinct \c Codec types and thus separate registries.
    struct Codec {
      virtual ~Codec() = default;
      virtual SerRet serialize(const AnalysisResult &, SerArgs...) const = 0;
      virtual DesRet deserialize(DesArgs...) const = 0;
    };

    template <class AnalysisResultT> struct Add {
      using TypedSerializerFn = SerRet (*)(const AnalysisResultT &, SerArgs...);

      static inline TypedSerializerFn SavedSerialize;
      static inline DeserializerFn SavedDeserialize;

      /// Takes the plugin's typed serializer and the deserializer, and
      /// inserts them into \c llvm::Registry<Codec>.
      Add(TypedSerializerFn TypedSerialize, DeserializerFn Deserialize) {
        /// Per-\c AnalysisResultT guard: each template instantiation gets
        /// its own \c static \c bool, so double-registration of the same
        /// analysis is caught even across translation units.
        static bool Registered = false;
        if (Registered) {
          ErrorBuilder::fatal("support is already registered for analysis: {0}",
                              AnalysisResultT::analysisName());
        }
        Registered = true;

        /// The plugin's serializer and deserializer are captured in
        /// static inline members of the Add template so that the
        /// \c ConcreteCodec default constructor (required by \c llvm::Registry)
        /// can read them. They use raw function pointers to prevent dangling
        /// references to temporary stack variables during registration.
        ///
        /// Once read by the constructor, they are stored as instance members
        /// of \c ConcreteCodec rather than directly executed from the \c static
        /// \c inline class members. This prevents symbol visibility issues
        /// across shared library boundaries on Linux (where \c dlopen with \c
        /// RTLD_LOCAL can give the host and plugin separate copies of \c static
        /// \c inline members).
        SavedSerialize = TypedSerialize;
        SavedDeserialize = Deserialize;

        /// Concrete subclass of \c Codec for \c AnalysisResultT.
        /// The \c serialize() override performs the downcast from
        /// \c AnalysisResult to \c AnalysisResultT.
        struct ConcreteCodec final : Codec {
          TypedSerializerFn SerFn;
          DeserializerFn DesFn;

          ConcreteCodec() : SerFn(SavedSerialize), DesFn(SavedDeserialize) {}

          SerRet serialize(const AnalysisResult &Base,
                           SerArgs... args) const override {
            return SerFn(static_cast<const AnalysisResultT &>(Base), args...);
          }

          DesRet deserialize(DesArgs... args) const override {
            return DesFn(args...);
          }
        };

        /// \c llvm::Registry stores the name as a \c StringRef, so the
        /// underlying string must be kept alive with a static declaration.
        static std::string NameStr =
            AnalysisResultT::analysisName().str().str();

        /// This performs the actual registration. It appends a factory for \c
        /// ConcreteCodec to the global \c llvm::Registry<Codec>. \c static
        /// ensures the `Registry::Add` object lives for the entire program,
        /// keeping its codec and node alive in the registry's linked list.
        [[maybe_unused]] static
            typename llvm::Registry<Codec>::template Add<ConcreteCodec>
                RegisterUsingCtorSideEffect(NameStr, "");
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

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_SERIALIZATION_SERIALIZATIONFORMAT_H
