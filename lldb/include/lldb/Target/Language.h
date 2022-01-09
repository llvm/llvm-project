//===-- Language.h ---------------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_LANGUAGE_H
#define LLDB_TARGET_LANGUAGE_H

#include <functional>
#include <memory>
#include <set>
#include <vector>

#include "lldb/Core/Highlighter.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/DataFormatters/DumpValueObjectOptions.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/lldb-private.h"
#include "lldb/lldb-public.h"

namespace lldb_private {

class Language : public PluginInterface {
public:
  class TypeScavenger {
  public:
    class Result {
    public:
      virtual bool IsValid() = 0;

      virtual bool DumpToStream(Stream &stream,
                                bool print_help_if_available) = 0;

      virtual ~Result() = default;
    };

    typedef std::set<std::unique_ptr<Result>> ResultSet;

    virtual ~TypeScavenger() = default;

    size_t Find(ExecutionContextScope *exe_scope, const char *key,
                ResultSet &results, bool append = true);

  protected:
    TypeScavenger() = default;

    virtual bool Find_Impl(ExecutionContextScope *exe_scope, const char *key,
                           ResultSet &results) = 0;
  };

  class ImageListTypeScavenger : public TypeScavenger {
    class Result : public Language::TypeScavenger::Result {
    public:
      Result(CompilerType type) : m_compiler_type(type) {}

      bool IsValid() override { return m_compiler_type.IsValid(); }

      bool DumpToStream(Stream &stream, bool print_help_if_available) override {
        if (IsValid()) {
          m_compiler_type.DumpTypeDescription(&stream);
          stream.EOL();
          return true;
        }
        return false;
      }

      ~Result() override = default;

    private:
      CompilerType m_compiler_type;
    };

  protected:
    ImageListTypeScavenger() = default;

    ~ImageListTypeScavenger() override = default;

    // is this type something we should accept? it's usually going to be a
    // filter by language + maybe some sugar tweaking
    // returning an empty type means rejecting this candidate entirely;
    // any other result will be accepted as a valid match
    virtual CompilerType AdjustForInclusion(CompilerType &candidate) = 0;

    bool Find_Impl(ExecutionContextScope *exe_scope, const char *key,
                   ResultSet &results) override;
  };

  template <typename... ScavengerTypes>
  class EitherTypeScavenger : public TypeScavenger {
  public:
    EitherTypeScavenger() : TypeScavenger() {
      for (std::shared_ptr<TypeScavenger> scavenger : { std::shared_ptr<TypeScavenger>(new ScavengerTypes())... }) {
        if (scavenger)
          m_scavengers.push_back(scavenger);
      }
    }
  protected:
    bool Find_Impl(ExecutionContextScope *exe_scope, const char *key,
                   ResultSet &results) override {
      const bool append = false;
      for (auto& scavenger : m_scavengers) {
        if (scavenger && scavenger->Find(exe_scope, key, results, append))
          return true;
      }
      return false;
    }
  private:
    std::vector<std::shared_ptr<TypeScavenger>> m_scavengers;
  };

  template <typename... ScavengerTypes>
  class UnionTypeScavenger : public TypeScavenger {
  public:
    UnionTypeScavenger() : TypeScavenger() {
      for (std::shared_ptr<TypeScavenger> scavenger : { std::shared_ptr<TypeScavenger>(new ScavengerTypes())... }) {
        if (scavenger)
          m_scavengers.push_back(scavenger);
      }
    }
  protected:
    bool Find_Impl(ExecutionContextScope *exe_scope, const char *key,
                   ResultSet &results) override {
      const bool append = true;
      bool success = false;
      for (auto& scavenger : m_scavengers) {
        if (scavenger)
          success = scavenger->Find(exe_scope, key, results, append) || success;
      }
      return success;
    }
  private:
    std::vector<std::shared_ptr<TypeScavenger>> m_scavengers;
  };

  enum class FunctionNameRepresentation {
    eName,
    eNameWithArgs,
    eNameWithNoArgs
  };

  ~Language() override;

  static Language *FindPlugin(lldb::LanguageType language);

  /// Returns the Language associated with the given file path or a nullptr
  /// if there is no known language.
  static Language *FindPlugin(llvm::StringRef file_path);

  static Language *FindPlugin(lldb::LanguageType language,
                              llvm::StringRef file_path);

  // return false from callback to stop iterating
  static void ForEach(std::function<bool(Language *)> callback);

  virtual lldb::LanguageType GetLanguageType() const = 0;

  virtual bool IsTopLevelFunction(Function &function);

  virtual bool IsSourceFile(llvm::StringRef file_path) const = 0;

  virtual const Highlighter *GetHighlighter() const { return nullptr; }

  virtual lldb::TypeCategoryImplSP GetFormatters();

  virtual HardcodedFormatters::HardcodedFormatFinder GetHardcodedFormats();

  virtual HardcodedFormatters::HardcodedSummaryFinder GetHardcodedSummaries();

  virtual HardcodedFormatters::HardcodedSyntheticFinder
  GetHardcodedSynthetics();

  virtual std::vector<ConstString>
  GetPossibleFormattersMatches(ValueObject &valobj,
                               lldb::DynamicValueType use_dynamic);

  virtual std::unique_ptr<TypeScavenger> GetTypeScavenger();

  virtual const char *GetLanguageSpecificTypeLookupHelp();

  class MethodNameVariant {
    ConstString m_name;
    lldb::FunctionNameType m_type;

  public:
    MethodNameVariant(ConstString name, lldb::FunctionNameType type)
        : m_name(name), m_type(type) {}
    ConstString GetName() const { return m_name; }
    lldb::FunctionNameType GetType() const { return m_type; }
  };
  // If a language can have more than one possible name for a method, this
  // function can be used to enumerate them. This is useful when doing name
  // lookups.
  virtual std::vector<Language::MethodNameVariant>
  GetMethodNameVariants(ConstString method_name) const {
    return std::vector<Language::MethodNameVariant>();
  };

  /// Returns true iff the given symbol name is compatible with the mangling
  /// scheme of this language.
  ///
  /// This function should only return true if there is a high confidence
  /// that the name actually belongs to this language.
  virtual bool SymbolNameFitsToLanguage(Mangled name) const { return false; }

  // if an individual data formatter can apply to several types and cross a
  // language boundary it makes sense for individual languages to want to
  // customize the printing of values of that type by appending proper
  // prefix/suffix information in language-specific ways
  virtual bool GetFormatterPrefixSuffix(ValueObject &valobj,
                                        ConstString type_hint,
                                        std::string &prefix,
                                        std::string &suffix);

  // if a language has a custom format for printing variable declarations that
  // it wants LLDB to honor it should return an appropriate closure here
  virtual DumpValueObjectOptions::DeclPrintingHelper GetDeclPrintingHelper();

  virtual LazyBool IsLogicalTrue(ValueObject &valobj, Status &error);

  // for a ValueObject of some "reference type", if the value points to the
  // nil/null object, this method returns true
  virtual bool IsNilReference(ValueObject &valobj);

  /// Returns the summary string for ValueObjects for which IsNilReference() is
  /// true.
  virtual llvm::StringRef GetNilReferenceSummaryString() { return {}; }

  // for a ValueObject of some "reference type", if the language provides a
  // technique to decide whether the reference has ever been assigned to some
  // object, this method will return true if such detection is possible, and if
  // the reference has never been assigned
  virtual bool IsUninitializedReference(ValueObject &valobj);

  virtual bool GetFunctionDisplayName(const SymbolContext *sc,
                                      const ExecutionContext *exe_ctx,
                                      FunctionNameRepresentation representation,
                                      Stream &s);

  virtual ConstString
  GetDemangledFunctionNameWithoutArguments(Mangled mangled) const {
    if (ConstString demangled = mangled.GetDemangledName())
      return demangled;

    return mangled.GetMangledName();
  }

  virtual void GetExceptionResolverDescription(bool catch_on, bool throw_on,
                                               Stream &s);

  static void GetDefaultExceptionResolverDescription(bool catch_on,
                                                     bool throw_on, Stream &s);

  // These are accessors for general information about the Languages lldb knows
  // about:

  static lldb::LanguageType
  GetLanguageTypeFromString(const char *string) = delete;
  static lldb::LanguageType GetLanguageTypeFromString(llvm::StringRef string);

  static const char *GetNameForLanguageType(lldb::LanguageType language);

  static void PrintAllLanguages(Stream &s, const char *prefix,
                                const char *suffix);

  // return false from callback to stop iterating
  static void ForAllLanguages(std::function<bool(lldb::LanguageType)> callback);

  static bool LanguageIsCPlusPlus(lldb::LanguageType language);

  static bool LanguageIsObjC(lldb::LanguageType language);

  static bool LanguageIsC(lldb::LanguageType language);

  /// Equivalent to \c LanguageIsC||LanguageIsObjC||LanguageIsCPlusPlus.
  static bool LanguageIsCFamily(lldb::LanguageType language);

  static bool LanguageIsPascal(lldb::LanguageType language);

  // return the primary language, so if LanguageIsC(l), return eLanguageTypeC,
  // etc.
  static lldb::LanguageType GetPrimaryLanguage(lldb::LanguageType language);

  static std::set<lldb::LanguageType> GetSupportedLanguages();

  static LanguageSet GetLanguagesSupportingTypeSystems();
  static LanguageSet GetLanguagesSupportingTypeSystemsForExpressions();
  static LanguageSet GetLanguagesSupportingREPLs();

  // Given a mangled function name, calculates some alternative manglings since
  // the compiler mangling may not line up with the symbol we are expecting.
  virtual std::vector<ConstString>
  GenerateAlternateFunctionManglings(const ConstString mangled) const {
    return std::vector<ConstString>();
  }

  virtual ConstString
  FindBestAlternateFunctionMangledName(const Mangled mangled,
                                       const SymbolContext &sym_ctx) const {
    return ConstString();
  }

protected:
  // Classes that inherit from Language can see and modify these

  Language();

private:
  Language(const Language &) = delete;
  const Language &operator=(const Language &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_LANGUAGE_H
