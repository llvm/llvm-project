//===--- Driver.h - Clang GCC Compatible Driver -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_DRIVER_H_
#define CLANG_DRIVER_DRIVER_H_

#include "clang/Basic/Diagnostic.h"

#include "clang/Driver/Phases.h"
#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Path.h" // FIXME: Kill when CompilationInfo
                              // lands.
#include <list>
#include <set>
#include <string>

namespace llvm {
  template<typename T> class ArrayRef;
}
namespace clang {
namespace driver {
  class Action;
  class Arg;
  class ArgList;
  class Command;
  class Compilation;
  class DerivedArgList;
  class InputArgList;
  class InputInfo;
  class JobAction;
  class OptTable;
  class ToolChain;

/// Driver - Encapsulate logic for constructing compilation processes
/// from a set of gcc-driver-like command line arguments.
class Driver {
  OptTable *Opts;

  DiagnosticsEngine &Diags;

public:
  // Diag - Forwarding function for diagnostics.
  DiagnosticBuilder Diag(unsigned DiagID) const {
    return Diags.Report(DiagID);
  }

  // FIXME: Privatize once interface is stable.
public:
  /// The name the driver was invoked as.
  std::string Name;

  /// The path the driver executable was in, as invoked from the
  /// command line.
  std::string Dir;

  /// The original path to the clang executable.
  std::string ClangExecutable;

  /// The path to the installed clang directory, if any.
  std::string InstalledDir;

  /// The path to the compiler resource directory.
  std::string ResourceDir;

  /// A prefix directory used to emulated a limited subset of GCC's '-Bprefix'
  /// functionality.
  /// FIXME: This type of customization should be removed in favor of the
  /// universal driver when it is ready.
  typedef SmallVector<std::string, 4> prefix_list;
  prefix_list PrefixDirs;

  /// sysroot, if present
  std::string SysRoot;

  /// If the standard library is used
  bool UseStdLib;

  /// Default target triple.
  std::string DefaultTargetTriple;

  /// Default name for linked images (e.g., "a.out").
  std::string DefaultImageName;

  /// Driver title to use with help.
  std::string DriverTitle;

  /// Information about the host which can be overridden by the user.
  std::string HostBits, HostMachine, HostSystem, HostRelease;

  /// The file to log CC_PRINT_OPTIONS output to, if enabled.
  const char *CCPrintOptionsFilename;

  /// The file to log CC_PRINT_HEADERS output to, if enabled.
  const char *CCPrintHeadersFilename;

  /// The file to log CC_LOG_DIAGNOSTICS output to, if enabled.
  const char *CCLogDiagnosticsFilename;

  /// A list of inputs and their types for the given arguments.
  typedef SmallVector<std::pair<types::ID, const Arg*>, 16> InputList;

  /// Whether the driver should follow g++ like behavior.
  unsigned CCCIsCXX : 1;

  /// Whether the driver is just the preprocessor.
  unsigned CCCIsCPP : 1;

  /// Echo commands while executing (in -v style).
  unsigned CCCEcho : 1;

  /// Only print tool bindings, don't build any jobs.
  unsigned CCCPrintBindings : 1;

  /// Set CC_PRINT_OPTIONS mode, which is like -v but logs the commands to
  /// CCPrintOptionsFilename or to stderr.
  unsigned CCPrintOptions : 1;

  /// Set CC_PRINT_HEADERS mode, which causes the frontend to log header include
  /// information to CCPrintHeadersFilename or to stderr.
  unsigned CCPrintHeaders : 1;

  /// Set CC_LOG_DIAGNOSTICS mode, which causes the frontend to log diagnostics
  /// to CCLogDiagnosticsFilename or to stderr, in a stable machine readable
  /// format.
  unsigned CCLogDiagnostics : 1;

  /// Whether the driver is generating diagnostics for debugging purposes.
  unsigned CCGenDiagnostics : 1;

private:
  /// Name to use when invoking gcc/g++.
  std::string CCCGenericGCCName;

  /// Whether to check that input files exist when constructing compilation
  /// jobs.
  unsigned CheckInputsExist : 1;

public:
  /// Use lazy precompiled headers for PCH support.
  unsigned CCCUsePCH : 1;

private:
  /// Certain options suppress the 'no input files' warning.
  bool SuppressMissingInputWarning : 1;

  std::list<std::string> TempFiles;
  std::list<std::string> ResultFiles;

  /// \brief Cache of all the ToolChains in use by the driver.
  ///
  /// This maps from the string representation of a triple to a ToolChain
  /// created targeting that triple. The driver owns all the ToolChain objects
  /// stored in it, and will clean them up when torn down.
  mutable llvm::StringMap<ToolChain *> ToolChains;

private:
  /// TranslateInputArgs - Create a new derived argument list from the input
  /// arguments, after applying the standard argument translations.
  DerivedArgList *TranslateInputArgs(const InputArgList &Args) const;

  // getFinalPhase - Determine which compilation mode we are in and record 
  // which option we used to determine the final phase.
  phases::ID getFinalPhase(const DerivedArgList &DAL, Arg **FinalPhaseArg = 0)
    const;

public:
  Driver(StringRef _ClangExecutable,
         StringRef _DefaultTargetTriple,
         StringRef _DefaultImageName,
         bool IsProduction,
         DiagnosticsEngine &_Diags);
  ~Driver();

  /// @name Accessors
  /// @{

  /// Name to use when invoking gcc/g++.
  const std::string &getCCCGenericGCCName() const { return CCCGenericGCCName; }


  const OptTable &getOpts() const { return *Opts; }

  const DiagnosticsEngine &getDiags() const { return Diags; }

  bool getCheckInputsExist() const { return CheckInputsExist; }

  void setCheckInputsExist(bool Value) { CheckInputsExist = Value; }

  const std::string &getTitle() { return DriverTitle; }
  void setTitle(std::string Value) { DriverTitle = Value; }

  /// \brief Get the path to the main clang executable.
  const char *getClangProgramPath() const {
    return ClangExecutable.c_str();
  }

  /// \brief Get the path to where the clang executable was installed.
  const char *getInstalledDir() const {
    if (!InstalledDir.empty())
      return InstalledDir.c_str();
    return Dir.c_str();
  }
  void setInstalledDir(StringRef Value) {
    InstalledDir = Value;
  }

  /// @}
  /// @name Primary Functionality
  /// @{

  /// BuildCompilation - Construct a compilation object for a command
  /// line argument vector.
  ///
  /// \return A compilation, or 0 if none was built for the given
  /// argument vector. A null return value does not necessarily
  /// indicate an error condition, the diagnostics should be queried
  /// to determine if an error occurred.
  Compilation *BuildCompilation(ArrayRef<const char *> Args);

  /// @name Driver Steps
  /// @{

  /// ParseArgStrings - Parse the given list of strings into an
  /// ArgList.
  InputArgList *ParseArgStrings(ArrayRef<const char *> Args);

  /// BuildInputs - Construct the list of inputs and their types from 
  /// the given arguments.
  ///
  /// \param TC - The default host tool chain.
  /// \param Args - The input arguments.
  /// \param Inputs - The list to store the resulting compilation 
  /// inputs onto.
  void BuildInputs(const ToolChain &TC, const DerivedArgList &Args,
                   InputList &Inputs) const;

  /// BuildActions - Construct the list of actions to perform for the
  /// given arguments, which are only done for a single architecture.
  ///
  /// \param TC - The default host tool chain.
  /// \param Args - The input arguments.
  /// \param Actions - The list to store the resulting actions onto.
  void BuildActions(const ToolChain &TC, const DerivedArgList &Args,
                    const InputList &Inputs, ActionList &Actions) const;

  /// BuildUniversalActions - Construct the list of actions to perform
  /// for the given arguments, which may require a universal build.
  ///
  /// \param TC - The default host tool chain.
  /// \param Args - The input arguments.
  /// \param Actions - The list to store the resulting actions onto.
  void BuildUniversalActions(const ToolChain &TC, const DerivedArgList &Args,
                             const InputList &BAInputs,
                             ActionList &Actions) const;

  /// BuildJobs - Bind actions to concrete tools and translate
  /// arguments to form the list of jobs to run.
  ///
  /// \param C - The compilation that is being built.
  void BuildJobs(Compilation &C) const;

  /// ExecuteCompilation - Execute the compilation according to the command line
  /// arguments and return an appropriate exit code.
  ///
  /// This routine handles additional processing that must be done in addition
  /// to just running the subprocesses, for example reporting errors, removing
  /// temporary files, etc.
  int ExecuteCompilation(const Compilation &C,
                         const Command *&FailingCommand) const;
  
  /// generateCompilationDiagnostics - Generate diagnostics information 
  /// including preprocessed source file(s).
  /// 
  void generateCompilationDiagnostics(Compilation &C,
                                      const Command *FailingCommand);

  /// @}
  /// @name Helper Methods
  /// @{

  /// PrintActions - Print the list of actions.
  void PrintActions(const Compilation &C) const;

  /// PrintHelp - Print the help text.
  ///
  /// \param ShowHidden - Show hidden options.
  void PrintHelp(bool ShowHidden) const;

  /// PrintOptions - Print the list of arguments.
  void PrintOptions(const ArgList &Args) const;

  /// PrintVersion - Print the driver version.
  void PrintVersion(const Compilation &C, raw_ostream &OS) const;

  /// GetFilePath - Lookup \p Name in the list of file search paths.
  ///
  /// \param TC - The tool chain for additional information on
  /// directories to search.
  //
  // FIXME: This should be in CompilationInfo.
  std::string GetFilePath(const char *Name, const ToolChain &TC) const;

  /// GetProgramPath - Lookup \p Name in the list of program search paths.
  ///
  /// \param TC - The provided tool chain for additional information on
  /// directories to search.
  //
  // FIXME: This should be in CompilationInfo.
  std::string GetProgramPath(const char *Name, const ToolChain &TC) const;

  /// HandleImmediateArgs - Handle any arguments which should be
  /// treated before building actions or binding tools.
  ///
  /// \return Whether any compilation should be built for this
  /// invocation.
  bool HandleImmediateArgs(const Compilation &C);

  /// ConstructAction - Construct the appropriate action to do for
  /// \p Phase on the \p Input, taking in to account arguments
  /// like -fsyntax-only or --analyze.
  Action *ConstructPhaseAction(const ArgList &Args, phases::ID Phase,
                               Action *Input) const;


  /// BuildJobsForAction - Construct the jobs to perform for the
  /// action \p A.
  void BuildJobsForAction(Compilation &C,
                          const Action *A,
                          const ToolChain *TC,
                          const char *BoundArch,
                          bool AtTopLevel,
                          const char *LinkingOutput,
                          InputInfo &Result) const;

  /// GetNamedOutputPath - Return the name to use for the output of
  /// the action \p JA. The result is appended to the compilation's
  /// list of temporary or result files, as appropriate.
  ///
  /// \param C - The compilation.
  /// \param JA - The action of interest.
  /// \param BaseInput - The original input file that this action was
  /// triggered by.
  /// \param AtTopLevel - Whether this is a "top-level" action.
  const char *GetNamedOutputPath(Compilation &C,
                                 const JobAction &JA,
                                 const char *BaseInput,
                                 bool AtTopLevel) const;

  /// GetTemporaryPath - Return the pathname of a temporary file to use 
  /// as part of compilation; the file will have the given prefix and suffix.
  ///
  /// GCC goes to extra lengths here to be a bit more robust.
  std::string GetTemporaryPath(StringRef Prefix, const char *Suffix) const;

  /// ShouldUseClangCompilar - Should the clang compiler be used to
  /// handle this action.
  bool ShouldUseClangCompiler(const Compilation &C, const JobAction &JA,
                              const llvm::Triple &ArchName) const;

  bool IsUsingLTO(const ArgList &Args) const;

private:
  /// \brief Retrieves a ToolChain for a particular target triple.
  ///
  /// Will cache ToolChains for the life of the driver object, and create them
  /// on-demand.
  const ToolChain &getToolChain(const ArgList &Args,
                                StringRef DarwinArchName = "") const;

  /// @}

public:
  /// GetReleaseVersion - Parse (([0-9]+)(.([0-9]+)(.([0-9]+)?))?)? and
  /// return the grouped values as integers. Numbers which are not
  /// provided are set to 0.
  ///
  /// \return True if the entire string was parsed (9.2), or all
  /// groups were parsed (10.3.5extrastuff). HadExtra is true if all
  /// groups were parsed but extra characters remain at the end.
  static bool GetReleaseVersion(const char *Str, unsigned &Major,
                                unsigned &Minor, unsigned &Micro,
                                bool &HadExtra);
};

} // end namespace driver
} // end namespace clang

#endif
