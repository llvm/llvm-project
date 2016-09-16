//===-- Language.h ---------------------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Language_h_
#define liblldb_Language_h_

// C Includes
// C++ Includes
#include <functional>
#include <memory>
#include <set>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/PluginInterface.h"
#include "lldb/DataFormatters/DumpValueObjectOptions.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/StringPrinter.h"
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

  enum class FunctionNameRepresentation {
    eName,
    eNameWithArgs,
    eNameWithNoArgs
  };

  ~Language() override;

  static Language *FindPlugin(lldb::LanguageType language);

  // return false from callback to stop iterating
  static void ForEach(std::function<bool(Language *)> callback);

  virtual lldb::LanguageType GetLanguageType() const = 0;

  virtual bool IsTopLevelFunction(Function &function);

  virtual lldb::TypeCategoryImplSP GetFormatters();

  virtual HardcodedFormatters::HardcodedFormatFinder GetHardcodedFormats();

  virtual HardcodedFormatters::HardcodedSummaryFinder GetHardcodedSummaries();

  virtual HardcodedFormatters::HardcodedSyntheticFinder
  GetHardcodedSynthetics();

  virtual HardcodedFormatters::HardcodedValidatorFinder
  GetHardcodedValidators();

  virtual std::vector<ConstString>
  GetPossibleFormattersMatches(ValueObject &valobj,
                               lldb::DynamicValueType use_dynamic);

  virtual lldb_private::formatters::StringPrinter::EscapingHelper
      GetStringPrinterEscapingHelper(
          lldb_private::formatters::StringPrinter::GetPrintableElementType);

  virtual std::unique_ptr<TypeScavenger> GetTypeScavenger();

  virtual const char *GetLanguageSpecificTypeLookupHelp();

  // if an individual data formatter can apply to several types and cross a
  // language boundary
  // it makes sense for individual languages to want to customize the printing
  // of values of that
  // type by appending proper prefix/suffix information in language-specific
  // ways
  virtual bool GetFormatterPrefixSuffix(ValueObject &valobj,
                                        ConstString type_hint,
                                        std::string &prefix,
                                        std::string &suffix);

  // if a language has a custom format for printing variable declarations that
  // it wants LLDB to honor
  // it should return an appropriate closure here
  virtual DumpValueObjectOptions::DeclPrintingHelper GetDeclPrintingHelper();

  virtual LazyBool IsLogicalTrue(ValueObject &valobj, Error &error);

  // for a ValueObject of some "reference type", if the value points to the
  // nil/null object, this method returns true
  virtual bool IsNilReference(ValueObject &valobj);

  // for a ValueObject of some "reference type", if the language provides a
  // technique
  // to decide whether the reference has ever been assigned to some object, this
  // method
  // will return true if such detection is possible, and if the reference has
  // never been assigned
  virtual bool IsUninitializedReference(ValueObject &valobj);

  virtual bool GetFunctionDisplayName(const SymbolContext *sc,
                                      const ExecutionContext *exe_ctx,
                                      FunctionNameRepresentation representation,
                                      Stream &s);

  virtual void GetExceptionResolverDescription(bool catch_on, bool throw_on,
                                               Stream &s);

  static void GetDefaultExceptionResolverDescription(bool catch_on,
                                                     bool throw_on, Stream &s);

  // These are accessors for general information about the Languages lldb knows
  // about:

  // TODO: Convert this to using a StringRef.
  static lldb::LanguageType GetLanguageTypeFromString(const char *string);

  static const char *GetNameForLanguageType(lldb::LanguageType language);

  static void PrintAllLanguages(Stream &s, const char *prefix,
                                const char *suffix);

  // return false from callback to stop iterating
  static void ForAllLanguages(std::function<bool(lldb::LanguageType)> callback);

  static bool LanguageIsCPlusPlus(lldb::LanguageType language);

  static bool LanguageIsObjC(lldb::LanguageType language);

  static bool LanguageIsC(lldb::LanguageType language);

  static bool LanguageIsPascal(lldb::LanguageType language);

  // return the primary language, so if LanguageIsC(l), return eLanguageTypeC,
  // etc.
  static lldb::LanguageType GetPrimaryLanguage(lldb::LanguageType language);

  static void GetLanguagesSupportingTypeSystems(
      std::set<lldb::LanguageType> &languages,
      std::set<lldb::LanguageType> &languages_for_expressions);

  static void
  GetLanguagesSupportingREPLs(std::set<lldb::LanguageType> &languages);

protected:
  //------------------------------------------------------------------
  // Classes that inherit from Language can see and modify these
  //------------------------------------------------------------------

  Language();

private:
  DISALLOW_COPY_AND_ASSIGN(Language);
};

} // namespace lldb_private

#endif // liblldb_Language_h_
