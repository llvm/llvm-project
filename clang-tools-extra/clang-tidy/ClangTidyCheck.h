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
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringExtras.h"
#include <memory>
#include <type_traits>
#include <vector>

namespace clang {

class CompilerInstance;

namespace tidy {

/// \brief Base class for all clang-tidy checks.
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
  /// \brief Initializes the check with \p CheckName and \p Context.
  ///
  /// Derived classes must implement the constructor with this signature or
  /// delegate it. If a check needs to read options, it can do this in the
  /// constructor using the Options.get() methods below.
  ClangTidyCheck(StringRef CheckName, ClangTidyContext *Context);

  /// DEPRECATED: Use the other overload.
  virtual void registerPPCallbacks(CompilerInstance &Compiler) {}

  /// \brief Override this to register ``PPCallbacks`` in the preprocessor.
  ///
  /// This should be used for clang-tidy checks that analyze preprocessor-
  /// dependent properties, e.g. include directives and macro definitions.
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

  /// \brief Override this to register AST matchers with \p Finder.
  ///
  /// This should be used by clang-tidy checks that analyze code properties that
  /// dependent on AST knowledge.
  ///
  /// You can register as many matchers as necessary with \p Finder. Usually,
  /// "this" will be used as callback, but you can also specify other callback
  /// classes. Thereby, different matchers can trigger different callbacks.
  ///
  /// If you need to merge information between the different matchers, you can
  /// store these as members of the derived class. However, note that all
  /// matches occur in the order of the AST traversal.
  virtual void registerMatchers(ast_matchers::MatchFinder *Finder) {}

  /// \brief ``ClangTidyChecks`` that register ASTMatchers should do the actual
  /// work in here.
  virtual void check(const ast_matchers::MatchFinder::MatchResult &Result) {}

  /// \brief Add a diagnostic with the check's name.
  DiagnosticBuilder diag(SourceLocation Loc, StringRef Description,
                         DiagnosticIDs::Level Level = DiagnosticIDs::Warning);

  /// \brief Should store all options supported by this check with their
  /// current values or default values for options that haven't been overridden.
  ///
  /// The check should use ``Options.store()`` to store each option it supports
  /// whether it has the default value or it has been overridden.
  virtual void storeOptions(ClangTidyOptions::OptionMap &Options) {}

private:
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  StringRef getID() const override { return CheckName; }
  std::string CheckName;
  ClangTidyContext *Context;

protected:
  /// \brief Provides access to the ``ClangTidyCheck`` options via check-local
  /// names.
  ///
  /// Methods of this class prepend ``CheckName + "."`` to translate check-local
  /// option names to global option names.
  class OptionsView {
  public:
    /// \brief Initializes the instance using \p CheckName + "." as a prefix.
    OptionsView(StringRef CheckName,
                const ClangTidyOptions::OptionMap &CheckOptions);

    /// \brief Read a named option from the ``Context``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, returns
    /// \p Default.
    std::string get(StringRef LocalName, StringRef Default) const;

    /// \brief Read a named option from the ``Context``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, returns Default.
    std::string getLocalOrGlobal(StringRef LocalName, StringRef Default) const;

    /// \brief Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, returns
    /// \p Default.
    template <typename T>
    typename std::enable_if<std::is_integral<T>::value, T>::type
    get(StringRef LocalName, T Default) const {
      std::string Value = get(LocalName, "");
      T Result = Default;
      if (!Value.empty())
        StringRef(Value).getAsInteger(10, Result);
      return Result;
    }

    /// \brief Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, returns Default.
    template <typename T>
    typename std::enable_if<std::is_integral<T>::value, T>::type
    getLocalOrGlobal(StringRef LocalName, T Default) const {
      std::string Value = getLocalOrGlobal(LocalName, "");
      T Result = Default;
      if (!Value.empty())
        StringRef(Value).getAsInteger(10, Result);
      return Result;
    }

    /// \brief Stores an option with the check-local name \p LocalName with
    /// string value \p Value to \p Options.
    void store(ClangTidyOptions::OptionMap &Options, StringRef LocalName,
               StringRef Value) const;

    /// \brief Stores an option with the check-local name \p LocalName with
    /// ``int64_t`` value \p Value to \p Options.
    void store(ClangTidyOptions::OptionMap &Options, StringRef LocalName,
               int64_t Value) const;

  private:
    std::string NamePrefix;
    const ClangTidyOptions::OptionMap &CheckOptions;
  };

  OptionsView Options;
  /// \brief Returns the main file name of the current translation unit.
  StringRef getCurrentMainFile() const { return Context->getCurrentFile(); }
  /// \brief Returns the language options from the context.
  LangOptions getLangOpts() const { return Context->getLangOpts(); }
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYCHECK_H
