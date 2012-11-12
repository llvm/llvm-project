//===--- OptTable.h - Option Table ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_OPTTABLE_H
#define CLANG_DRIVER_OPTTABLE_H

#include "clang/Basic/LLVM.h"
#include "clang/Driver/OptSpecifier.h"
#include "llvm/ADT/StringSet.h"

namespace clang {
namespace driver {
  class Arg;
  class ArgList;
  class InputArgList;
  class Option;

  /// \brief Provide access to the Option info table.
  ///
  /// The OptTable class provides a layer of indirection which allows Option
  /// instance to be created lazily. In the common case, only a few options will
  /// be needed at runtime; the OptTable class maintains enough information to
  /// parse command lines without instantiating Options, while letting other
  /// parts of the driver still use Option instances where convenient.
  class OptTable {
  public:
    /// \brief Entry for a single option instance in the option data table.
    struct Info {
      /// A null terminated array of prefix strings to apply to name while
      /// matching.
      const char *const *Prefixes;
      const char *Name;
      const char *HelpText;
      const char *MetaVar;
      unsigned ID;
      unsigned char Kind;
      unsigned char Param;
      unsigned short Flags;
      unsigned short GroupID;
      unsigned short AliasID;
    };

  private:
    /// \brief The static option information table.
    const Info *OptionInfos;
    unsigned NumOptionInfos;

    unsigned TheInputOptionID;
    unsigned TheUnknownOptionID;

    /// The index of the first option which can be parsed (i.e., is not a
    /// special option like 'input' or 'unknown', and is not an option group).
    unsigned FirstSearchableIndex;

    /// The union of all option prefixes. If an argument does not begin with
    /// one of these, it is an input.
    llvm::StringSet<> PrefixesUnion;
    std::string PrefixChars;

  private:
    const Info &getInfo(OptSpecifier Opt) const {
      unsigned id = Opt.getID();
      assert(id > 0 && id - 1 < getNumOptions() && "Invalid Option ID.");
      return OptionInfos[id - 1];
    }

  protected:
    OptTable(const Info *_OptionInfos, unsigned _NumOptionInfos);
  public:
    ~OptTable();

    /// \brief Return the total number of option classes.
    unsigned getNumOptions() const { return NumOptionInfos; }

    /// \brief Get the given Opt's Option instance, lazily creating it
    /// if necessary.
    ///
    /// \return The option, or null for the INVALID option id.
    const Option getOption(OptSpecifier Opt) const;

    /// \brief Lookup the name of the given option.
    const char *getOptionName(OptSpecifier id) const {
      return getInfo(id).Name;
    }

    /// \brief Get the kind of the given option.
    unsigned getOptionKind(OptSpecifier id) const {
      return getInfo(id).Kind;
    }

    /// \brief Get the group id for the given option.
    unsigned getOptionGroupID(OptSpecifier id) const {
      return getInfo(id).GroupID;
    }

    /// \brief Get the help text to use to describe this option.
    const char *getOptionHelpText(OptSpecifier id) const {
      return getInfo(id).HelpText;
    }

    /// \brief Get the meta-variable name to use when describing
    /// this options values in the help text.
    const char *getOptionMetaVar(OptSpecifier id) const {
      return getInfo(id).MetaVar;
    }

    /// \brief Parse a single argument; returning the new argument and
    /// updating Index.
    ///
    /// \param [in,out] Index - The current parsing position in the argument
    /// string list; on return this will be the index of the next argument
    /// string to parse.
    ///
    /// \return The parsed argument, or 0 if the argument is missing values
    /// (in which case Index still points at the conceptual next argument string
    /// to parse).
    Arg *ParseOneArg(const ArgList &Args, unsigned &Index) const;

    /// \brief Parse an list of arguments into an InputArgList.
    ///
    /// The resulting InputArgList will reference the strings in [\p ArgBegin,
    /// \p ArgEnd), and their lifetime should extend past that of the returned
    /// InputArgList.
    ///
    /// The only error that can occur in this routine is if an argument is
    /// missing values; in this case \p MissingArgCount will be non-zero.
    ///
    /// \param ArgBegin - The beginning of the argument vector.
    /// \param ArgEnd - The end of the argument vector.
    /// \param MissingArgIndex - On error, the index of the option which could
    /// not be parsed.
    /// \param MissingArgCount - On error, the number of missing options.
    /// \return An InputArgList; on error this will contain all the options
    /// which could be parsed.
    InputArgList *ParseArgs(const char* const *ArgBegin,
                            const char* const *ArgEnd,
                            unsigned &MissingArgIndex,
                            unsigned &MissingArgCount) const;

    /// \brief Render the help text for an option table.
    ///
    /// \param OS - The stream to write the help text to.
    /// \param Name - The name to use in the usage line.
    /// \param Title - The title to use in the usage line.
    /// \param FlagsToInclude - If non-zero, only include options with any
    ///                         of these flags set.
    /// \param FlagsToExclude - Exclude options with any of these flags set.
    void PrintHelp(raw_ostream &OS, const char *Name,
                   const char *Title, unsigned short FlagsToInclude = 0,
                   unsigned short FlagsToExclude = 0) const;
  };
}
}

#endif
