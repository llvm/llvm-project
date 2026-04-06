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
  /// AnalysisResult internally.
  /// 2. \c llvm::Registry only stores nullary factories — there is no way
  ///    to pass constructor arguments when a registry entry is instantiated.
  ///    This forces the \c ConcreteEntry subclass + function-local statics
  ///    pattern: the statics capture the serializer/deserializer at
  ///    registration time, and \c ConcreteEntry's default constructor reads
  ///    them back when the registry later invokes the factory.
  ///
  /// \tparam FormatT Phantom type is needed to disambiguate \c llvm::Registry
  ///   instantiations. \c llvm::Registry is keyed on the \c Entry type,
  ///   so two formats sharing the same serializer/deserializer signatures
  ///   would collide on the same registry without this parameter.
  template <class FormatT, class SerializerFn, class DeserializerFn>
  class AnalysisResultRegistryGenerator {
  public:
    /// Base type stored in \c llvm::Registry<Entry>. Each registered
    /// analysis produces a subclass (\c ConcreteEntry) whose default
    /// constructor populates these fields.
    struct Entry {
      explicit Entry(SerializerFn Serialize, DeserializerFn Deserialize)
          : Serialize(Serialize), Deserialize(Deserialize) {}
      /// \c instantiate() creates a \c ConcreteEntry but returns it as
      /// \c unique_ptr<Entry>; deleting the derived object through the
      /// base pointer requires a virtual destructor to avoid UB.
      virtual ~Entry() = default;
      SerializerFn Serialize;
      DeserializerFn Deserialize;
    };

    template <class AnalysisResultT> struct Add {
      /// This class provides a wrap method that mediates the type conversion of
      /// the value to be serialized. A plugin author writes a serializer that
      /// take a concrete \c AnalysisResultT for serialization. However, the
      /// registry stores a \c function_ref<...(const AnalysisResult &, ...)>
      /// because the use site that invokes serialization is unaware of the
      /// concrete subtype \c AnalysisResultT; it only knows that it has an
      /// `AnalysisResult` value that needs to be serialized. The wrapper
      /// function below bridges this mismatch by performing a `static_cast`
      /// from `AnalysisResult` to `AnalysisResultT`. This conversion can only
      /// happen at the two places where we know the concrete `AnalysisResultT`:
      /// first is this struct which receives the concrete type as a template
      /// parameter; second is the user's serialization function itself, but
      /// that would require the user to cast explicitly.
      template <class> struct SerializerAdapter;
      template <class R, class... Args>
      struct SerializerAdapter<
          llvm::function_ref<R(const AnalysisResult &, Args...)>> {

        /// The function-pointer type the plugin author actually writes.
        using TypedFnPtr = R (*)(const AnalysisResultT &, Args...);

        /// Holds the plugin's typed function pointer so that \c wrap() can
        /// call it. \c static \c inline gives it program lifetime,
        /// independent of any single \c Add instance.
        static inline TypedFnPtr Saved = nullptr;

        /// Accepts the base \c AnalysisResult reference from the use site,
        /// downcasts to the concrete type, and forwards to the plugin's typed
        /// serializer stored in \c Saved.
        static R wrap(const AnalysisResult &Base, Args... args) {
          return Saved(static_cast<const AnalysisResultT &>(Base), args...);
        }
      };
      using SA = SerializerAdapter<SerializerFn>;

      /// Takes the plugin's typed serializer and the deserializer, and
      /// inserts them into \c llvm::Registry<Entry>.
      Add(typename SA::TypedFnPtr TypedSerialize, DeserializerFn Deserialize) {
        /// Per-\c AnalysisResultT guard: each template instantiation gets
        /// its own \c static \c bool, so double-registration of the same
        /// analysis is caught even across translation units.
        static bool Registered = false;
        if (Registered) {
          ErrorBuilder::fatal("support is already registered for analysis: {0}",
                              AnalysisResultT::analysisName());
        }
        Registered = true;

        SA::Saved = TypedSerialize;

        struct ConcreteEntry : Entry {
          ConcreteEntry() : Entry(SA::wrap, Deserialize) {}
        };

        /// \c llvm::Registry stores the name as a \c StringRef, so the
        /// underlying string must be kept alive with a static declaration.
        static std::string NameStr =
            AnalysisResultT::analysisName().str().str();

        /// This performs the actual registration. It appends a factory for \c
        /// ConcreteEntry to the global \c llvm::Registry<Entry> list. \c static
        /// ensures the `Registry::Add` object lives for the entire program,
        /// keeping its entry and node alive in the registry's linked list.
        [[maybe_unused]] static
            typename llvm::Registry<Entry>::template Add<ConcreteEntry>
                RegisterUsingCtorSideEffect(NameStr, "");
      }
    };

    /// Looks up the serializer/deserializer pair for \p Name by walking the
    /// registry list.
    static llvm::Expected<std::pair<SerializerFn, DeserializerFn>>
    instantiate(const AnalysisName &Name) {
      for (const auto &E : llvm::Registry<Entry>::entries()) {
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
