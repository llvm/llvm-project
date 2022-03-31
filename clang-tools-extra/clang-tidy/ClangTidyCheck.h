//===--- ClangTidyCheck.h - clang-tidy --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYCHECK_H

#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyOptions.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/Optional.h"
#include <type_traits>
#include <utility>
#include <vector>

namespace clang {

class SourceManager;

namespace tidy {

/// This class should be specialized by any enum type that needs to be converted
/// to and from an \ref llvm::StringRef.
template <class T> struct OptionEnumMapping {
  // Specializations of this struct must implement this function.
  static ArrayRef<std::pair<T, StringRef>> getEnumMapping() = delete;
};

/// Base class for all clang-tidy checks.
///
/// To implement a ``ClangTidyCheck``, write a subclass and override some of the
/// base class's methods. E.g. to implement a check that validates namespace
/// declarations, override ``registerMatchers``:
///
/// ~~~{.cpp}
/// void registerMatchers(ast_matchers::MatchFinder *Finder) override {
///   Finder->addMatcher(namespaceDecl().bind("namespace"), this);
/// }
/// ~~~
///
/// and then override ``check(const MatchResult &Result)`` to do the actual
/// check for each match.
///
/// A new ``ClangTidyCheck`` instance is created per translation unit.
///
/// FIXME: Figure out whether carrying information from one TU to another is
/// useful/necessary.
class ClangTidyCheck : public ast_matchers::MatchFinder::MatchCallback {
public:
  /// Initializes the check with \p CheckName and \p Context.
  ///
  /// Derived classes must implement the constructor with this signature or
  /// delegate it. If a check needs to read options, it can do this in the
  /// constructor using the Options.get() methods below.
  ClangTidyCheck(StringRef CheckName, ClangTidyContext *Context);

  /// Override this to disable registering matchers and PP callbacks if an
  /// invalid language version is being used.
  ///
  /// For example if a check is examining overloaded functions then this should
  /// be overridden to return false when the CPlusPlus flag is not set in
  /// \p LangOpts.
  virtual bool isLanguageVersionSupported(const LangOptions &LangOpts) const {
    return true;
  }

  /// Override this to register ``PPCallbacks`` in the preprocessor.
  ///
  /// This should be used for clang-tidy checks that analyze preprocessor-
  /// dependent properties, e.g. include directives and macro definitions.
  ///
  /// This will only be executed if the function isLanguageVersionSupported
  /// returns true.
  ///
  /// There are two Preprocessors to choose from that differ in how they handle
  /// modular #includes:
  ///  - PP is the real Preprocessor. It doesn't walk into modular #includes and
  ///    thus doesn't generate PPCallbacks for their contents.
  ///  - ModuleExpanderPP preprocesses the whole translation unit in the
  ///    non-modular mode, which allows it to generate PPCallbacks not only for
  ///    the main file and textual headers, but also for all transitively
  ///    included modular headers when the analysis runs with modules enabled.
  ///    When modules are not enabled ModuleExpanderPP just points to the real
  ///    preprocessor.
  virtual void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                                   Preprocessor *ModuleExpanderPP) {}

  /// Override this to register AST matchers with \p Finder.
  ///
  /// This should be used by clang-tidy checks that analyze code properties that
  /// dependent on AST knowledge.
  ///
  /// You can register as many matchers as necessary with \p Finder. Usually,
  /// "this" will be used as callback, but you can also specify other callback
  /// classes. Thereby, different matchers can trigger different callbacks.
  ///
  /// This will only be executed if the function isLanguageVersionSupported
  /// returns true.
  ///
  /// If you need to merge information between the different matchers, you can
  /// store these as members of the derived class. However, note that all
  /// matches occur in the order of the AST traversal.
  virtual void registerMatchers(ast_matchers::MatchFinder *Finder) {}

  /// ``ClangTidyChecks`` that register ASTMatchers should do the actual
  /// work in here.
  virtual void check(const ast_matchers::MatchFinder::MatchResult &Result) {}

  /// Add a diagnostic with the check's name.
  DiagnosticBuilder diag(SourceLocation Loc, StringRef Description,
                         DiagnosticIDs::Level Level = DiagnosticIDs::Warning);

  /// Add a diagnostic with the check's name.
  DiagnosticBuilder diag(StringRef Description,
                         DiagnosticIDs::Level Level = DiagnosticIDs::Warning);

  /// Adds a diagnostic to report errors in the check's configuration.
  DiagnosticBuilder
  configurationDiag(StringRef Description,
                    DiagnosticIDs::Level Level = DiagnosticIDs::Warning) const;

  /// Should store all options supported by this check with their
  /// current values or default values for options that haven't been overridden.
  ///
  /// The check should use ``Options.store()`` to store each option it supports
  /// whether it has the default value or it has been overridden.
  virtual void storeOptions(ClangTidyOptions::OptionMap &Options) {}

  /// Provides access to the ``ClangTidyCheck`` options via check-local
  /// names.
  ///
  /// Methods of this class prepend ``CheckName + "."`` to translate check-local
  /// option names to global option names.
  class OptionsView {
    void diagnoseBadIntegerOption(const Twine &Lookup,
                                  StringRef Unparsed) const;
    void diagnoseBadBooleanOption(const Twine &Lookup,
                                  StringRef Unparsed) const;
    void diagnoseBadEnumOption(const Twine &Lookup, StringRef Unparsed,
                               StringRef Suggestion = StringRef()) const;

  public:
    /// Initializes the instance using \p CheckName + "." as a prefix.
    OptionsView(StringRef CheckName,
                const ClangTidyOptions::OptionMap &CheckOptions,
                ClangTidyContext *Context);

    /// Read a named option from the ``Context``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, return
    /// ``None``.
    llvm::Optional<std::string> get(StringRef LocalName) const;

    /// Read a named option from the ``Context``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, returns
    /// \p Default.
    std::string get(StringRef LocalName, StringRef Default) const;

    /// Read a named option from the ``Context``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, return ``None``.
    llvm::Optional<std::string> getLocalOrGlobal(StringRef LocalName) const;

    /// Read a named option from the ``Context``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, returns \p Default.
    std::string getLocalOrGlobal(StringRef LocalName, StringRef Default) const;

    /// Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, return
    /// ``None``.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return ``None``.
    template <typename T>
    std::enable_if_t<std::is_integral<T>::value, llvm::Optional<T>>
    get(StringRef LocalName) const {
      if (llvm::Optional<std::string> Value = get(LocalName)) {
        T Result{};
        if (!StringRef(*Value).getAsInteger(10, Result))
          return Result;
        diagnoseBadIntegerOption(NamePrefix + LocalName, *Value);
      }
      return None;
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, return
    /// \p Default.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return \p Default.
    template <typename T>
    std::enable_if_t<std::is_integral<T>::value, T> get(StringRef LocalName,
                                                        T Default) const {
      return get<T>(LocalName).getValueOr(Default);
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, return ``None``.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return ``None``.
    template <typename T>
    std::enable_if_t<std::is_integral<T>::value, llvm::Optional<T>>
    getLocalOrGlobal(StringRef LocalName) const {
      llvm::Optional<std::string> ValueOr = get(LocalName);
      bool IsGlobal = false;
      if (!ValueOr) {
        IsGlobal = true;
        ValueOr = getLocalOrGlobal(LocalName);
        if (!ValueOr)
          return None;
      }
      T Result{};
      if (!StringRef(*ValueOr).getAsInteger(10, Result))
        return Result;
      diagnoseBadIntegerOption(
          IsGlobal ? Twine(LocalName) : NamePrefix + LocalName, *ValueOr);
      return None;
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, return \p Default.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return \p Default.
    template <typename T>
    std::enable_if_t<std::is_integral<T>::value, T>
    getLocalOrGlobal(StringRef LocalName, T Default) const {
      return getLocalOrGlobal<T>(LocalName).getValueOr(Default);
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// enum type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, return
    /// ``None``.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return ``None``.
    ///
    /// \ref clang::tidy::OptionEnumMapping must be specialized for ``T`` to
    /// supply the mapping required to convert between ``T`` and a string.
    template <typename T>
    std::enable_if_t<std::is_enum<T>::value, llvm::Optional<T>>
    get(StringRef LocalName, bool IgnoreCase = false) const {
      if (llvm::Optional<int64_t> ValueOr =
              getEnumInt(LocalName, typeEraseMapping<T>(), false, IgnoreCase))
        return static_cast<T>(*ValueOr);
      return None;
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// enum type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, return
    /// \p Default.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return \p Default.
    ///
    /// \ref clang::tidy::OptionEnumMapping must be specialized for ``T`` to
    /// supply the mapping required to convert between ``T`` and a string.
    template <typename T>
    std::enable_if_t<std::is_enum<T>::value, T>
    get(StringRef LocalName, T Default, bool IgnoreCase = false) const {
      return get<T>(LocalName, IgnoreCase).getValueOr(Default);
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// enum type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, returns ``None``.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return ``None``.
    ///
    /// \ref clang::tidy::OptionEnumMapping must be specialized for ``T`` to
    /// supply the mapping required to convert between ``T`` and a string.
    template <typename T>
    std::enable_if_t<std::is_enum<T>::value, llvm::Optional<T>>
    getLocalOrGlobal(StringRef LocalName, bool IgnoreCase = false) const {
      if (llvm::Optional<int64_t> ValueOr =
              getEnumInt(LocalName, typeEraseMapping<T>(), true, IgnoreCase))
        return static_cast<T>(*ValueOr);
      return None;
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// enum type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either return \p Default.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return \p Default.
    ///
    /// \ref clang::tidy::OptionEnumMapping must be specialized for ``T`` to
    /// supply the mapping required to convert between ``T`` and a string.
    template <typename T>
    std::enable_if_t<std::is_enum<T>::value, T>
    getLocalOrGlobal(StringRef LocalName, T Default,
                     bool IgnoreCase = false) const {
      return getLocalOrGlobal<T>(LocalName, IgnoreCase).getValueOr(Default);
    }

    /// Stores an option with the check-local name \p LocalName with
    /// string value \p Value to \p Options.
    void store(ClangTidyOptions::OptionMap &Options, StringRef LocalName,
               StringRef Value) const;

    /// Stores an option with the check-local name \p LocalName with
    /// integer value \p Value to \p Options.
    template <typename T>
    std::enable_if_t<std::is_integral<T>::value>
    store(ClangTidyOptions::OptionMap &Options, StringRef LocalName,
          T Value) const {
      storeInt(Options, LocalName, Value);
    }

    /// Stores an option with the check-local name \p LocalName as the string
    /// representation of the Enum \p Value to \p Options.
    ///
    /// \ref clang::tidy::OptionEnumMapping must be specialized for ``T`` to
    /// supply the mapping required to convert between ``T`` and a string.
    template <typename T>
    std::enable_if_t<std::is_enum<T>::value>
    store(ClangTidyOptions::OptionMap &Options, StringRef LocalName,
          T Value) const {
      ArrayRef<std::pair<T, StringRef>> Mapping =
          OptionEnumMapping<T>::getEnumMapping();
      auto Iter = llvm::find_if(
          Mapping, [&](const std::pair<T, StringRef> &NameAndEnum) {
            return NameAndEnum.first == Value;
          });
      assert(Iter != Mapping.end() && "Unknown Case Value");
      store(Options, LocalName, Iter->second);
    }

  private:
    using NameAndValue = std::pair<int64_t, StringRef>;

    llvm::Optional<int64_t> getEnumInt(StringRef LocalName,
                                       ArrayRef<NameAndValue> Mapping,
                                       bool CheckGlobal, bool IgnoreCase) const;

    template <typename T>
    std::enable_if_t<std::is_enum<T>::value, std::vector<NameAndValue>>
    typeEraseMapping() const {
      ArrayRef<std::pair<T, StringRef>> Mapping =
          OptionEnumMapping<T>::getEnumMapping();
      std::vector<NameAndValue> Result;
      Result.reserve(Mapping.size());
      for (auto &MappedItem : Mapping) {
        Result.emplace_back(static_cast<int64_t>(MappedItem.first),
                            MappedItem.second);
      }
      return Result;
    }

    void storeInt(ClangTidyOptions::OptionMap &Options, StringRef LocalName,
                  int64_t Value) const;


    std::string NamePrefix;
    const ClangTidyOptions::OptionMap &CheckOptions;
    ClangTidyContext *Context;
  };

private:
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  StringRef getID() const override { return CheckName; }
  std::string CheckName;
  ClangTidyContext *Context;

protected:
  OptionsView Options;
  /// Returns the main file name of the current translation unit.
  StringRef getCurrentMainFile() const { return Context->getCurrentFile(); }
  /// Returns the language options from the context.
  const LangOptions &getLangOpts() const { return Context->getLangOpts(); }
};

/// Read a named option from the ``Context`` and parse it as a bool.
///
/// Reads the option with the check-local name \p LocalName from the
/// ``CheckOptions``. If the corresponding key is not present, return
/// ``None``.
///
/// If the corresponding key can't be parsed as a bool, emit a
/// diagnostic and return ``None``.
template <>
llvm::Optional<bool>
ClangTidyCheck::OptionsView::get<bool>(StringRef LocalName) const;

/// Read a named option from the ``Context`` and parse it as a bool.
///
/// Reads the option with the check-local name \p LocalName from the
/// ``CheckOptions``. If the corresponding key is not present, return
/// \p Default.
///
/// If the corresponding key can't be parsed as a bool, emit a
/// diagnostic and return \p Default.
template <>
llvm::Optional<bool>
ClangTidyCheck::OptionsView::getLocalOrGlobal<bool>(StringRef LocalName) const;

/// Stores an option with the check-local name \p LocalName with
/// bool value \p Value to \p Options.
template <>
void ClangTidyCheck::OptionsView::store<bool>(
    ClangTidyOptions::OptionMap &Options, StringRef LocalName,
    bool Value) const;


} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYCHECK_H
