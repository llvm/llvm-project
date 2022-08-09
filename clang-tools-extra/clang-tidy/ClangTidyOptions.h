//===--- ClangTidyOptions.h - clang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYOPTIONS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYOPTIONS_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <functional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace clang {
namespace tidy {

/// Contains a list of line ranges in a single file.
struct FileFilter {
  /// File name.
  std::string Name;

  /// LineRange is a pair<start, end> (inclusive).
  typedef std::pair<unsigned, unsigned> LineRange;

  /// A list of line ranges in this file, for which we show warnings.
  std::vector<LineRange> LineRanges;
};

/// Global options. These options are neither stored nor read from
/// configuration files.
struct ClangTidyGlobalOptions {
  /// Output warnings from certain line ranges of certain files only.
  /// If empty, no warnings will be filtered.
  std::vector<FileFilter> LineFilter;
};

/// Contains options for clang-tidy. These options may be read from
/// configuration files, and may be different for different translation units.
struct ClangTidyOptions {
  /// These options are used for all settings that haven't been
  /// overridden by the \c OptionsProvider.
  ///
  /// Allow no checks and no headers by default. This method initializes
  /// check-specific options by calling \c ClangTidyModule::getModuleOptions()
  /// of each registered \c ClangTidyModule.
  static ClangTidyOptions getDefaults();

  /// Overwrites all fields in here by the fields of \p Other that have a value.
  /// \p Order specifies precedence of \p Other option.
  ClangTidyOptions &mergeWith(const ClangTidyOptions &Other, unsigned Order);

  /// Creates a new \c ClangTidyOptions instance combined from all fields
  /// of this instance overridden by the fields of \p Other that have a value.
  /// \p Order specifies precedence of \p Other option.
  [[nodiscard]] ClangTidyOptions merge(const ClangTidyOptions &Other,
                                       unsigned Order) const;

  /// Checks filter.
  llvm::Optional<std::string> Checks;

  /// WarningsAsErrors filter.
  llvm::Optional<std::string> WarningsAsErrors;

  /// Output warnings from headers matching this filter. Warnings from
  /// main files will always be displayed.
  llvm::Optional<std::string> HeaderFilterRegex;

  /// Output warnings from system headers matching \c HeaderFilterRegex.
  llvm::Optional<bool> SystemHeaders;

  /// Format code around applied fixes with clang-format using this
  /// style.
  ///
  /// Can be one of:
  ///   * 'none' - don't format code around applied fixes;
  ///   * 'llvm', 'google', 'mozilla' or other predefined clang-format style
  ///     names;
  ///   * 'file' - use the .clang-format file in the closest parent directory of
  ///     each source file;
  ///   * '{inline-formatting-style-in-yaml-format}'.
  ///
  /// See clang-format documentation for more about configuring format style.
  llvm::Optional<std::string> FormatStyle;

  /// Specifies the name or e-mail of the user running clang-tidy.
  ///
  /// This option is used, for example, to place the correct user name in TODO()
  /// comments in the relevant check.
  llvm::Optional<std::string> User;

  /// Helper structure for storing option value with priority of the value.
  struct ClangTidyValue {
    ClangTidyValue() : Value(), Priority(0) {}
    ClangTidyValue(const char *Value) : Value(Value), Priority(0) {}
    ClangTidyValue(llvm::StringRef Value, unsigned Priority = 0)
        : Value(Value), Priority(Priority) {}

    std::string Value;
    /// Priority stores relative precedence of the value loaded from config
    /// files to disambiguate local vs global value from different levels.
    unsigned Priority;
  };
  typedef std::pair<std::string, std::string> StringPair;
  typedef llvm::StringMap<ClangTidyValue> OptionMap;

  /// Key-value mapping used to store check-specific options.
  OptionMap CheckOptions;

  typedef std::vector<std::string> ArgList;

  /// Add extra compilation arguments to the end of the list.
  llvm::Optional<ArgList> ExtraArgs;

  /// Add extra compilation arguments to the start of the list.
  llvm::Optional<ArgList> ExtraArgsBefore;

  /// Only used in the FileOptionsProvider and ConfigOptionsProvider. If true
  /// and using a FileOptionsProvider, it will take a configuration file in the
  /// parent directory (if any exists) and apply this config file on top of the
  /// parent one. IF true and using a ConfigOptionsProvider, it will apply this
  /// config on top of any configuration file it finds in the directory using
  /// the same logic as FileOptionsProvider. If false or missing, only this
  /// configuration file will be used.
  llvm::Optional<bool> InheritParentConfig;

  /// Use colors in diagnostics. If missing, it will be auto detected.
  llvm::Optional<bool> UseColor;
};

/// Abstract interface for retrieving various ClangTidy options.
class ClangTidyOptionsProvider {
public:
  static const char OptionsSourceTypeDefaultBinary[];
  static const char OptionsSourceTypeCheckCommandLineOption[];
  static const char OptionsSourceTypeConfigCommandLineOption[];

  virtual ~ClangTidyOptionsProvider() {}

  /// Returns global options, which are independent of the file.
  virtual const ClangTidyGlobalOptions &getGlobalOptions() = 0;

  /// ClangTidyOptions and its source.
  //
  /// clang-tidy has 3 types of the sources in order of increasing priority:
  ///    * clang-tidy binary.
  ///    * '-config' commandline option or a specific configuration file. If the
  ///       commandline option is specified, clang-tidy will ignore the
  ///       configuration file.
  ///    * '-checks' commandline option.
  typedef std::pair<ClangTidyOptions, std::string> OptionsSource;

  /// Returns an ordered vector of OptionsSources, in order of increasing
  /// priority.
  virtual std::vector<OptionsSource>
  getRawOptions(llvm::StringRef FileName) = 0;

  /// Returns options applying to a specific translation unit with the
  /// specified \p FileName.
  ClangTidyOptions getOptions(llvm::StringRef FileName);
};

/// Implementation of the \c ClangTidyOptionsProvider interface, which
/// returns the same options for all files.
class DefaultOptionsProvider : public ClangTidyOptionsProvider {
public:
  DefaultOptionsProvider(ClangTidyGlobalOptions GlobalOptions,
                         ClangTidyOptions Options)
      : GlobalOptions(std::move(GlobalOptions)),
        DefaultOptions(std::move(Options)) {}
  const ClangTidyGlobalOptions &getGlobalOptions() override {
    return GlobalOptions;
  }
  std::vector<OptionsSource> getRawOptions(llvm::StringRef FileName) override;

private:
  ClangTidyGlobalOptions GlobalOptions;
  ClangTidyOptions DefaultOptions;
};

class FileOptionsBaseProvider : public DefaultOptionsProvider {
protected:
  // A pair of configuration file base name and a function parsing
  // configuration from text in the corresponding format.
  typedef std::pair<std::string, std::function<llvm::ErrorOr<ClangTidyOptions>(
                                     llvm::MemoryBufferRef)>>
      ConfigFileHandler;

  /// Configuration file handlers listed in the order of priority.
  ///
  /// Custom configuration file formats can be supported by constructing the
  /// list of handlers and passing it to the appropriate \c FileOptionsProvider
  /// constructor. E.g. initialization of a \c FileOptionsProvider with support
  /// of a custom configuration file format for files named ".my-tidy-config"
  /// could look similar to this:
  /// \code
  /// FileOptionsProvider::ConfigFileHandlers ConfigHandlers;
  /// ConfigHandlers.emplace_back(".my-tidy-config", parseMyConfigFormat);
  /// ConfigHandlers.emplace_back(".clang-tidy", parseConfiguration);
  /// return std::make_unique<FileOptionsProvider>(
  ///     GlobalOptions, DefaultOptions, OverrideOptions, ConfigHandlers);
  /// \endcode
  ///
  /// With the order of handlers shown above, the ".my-tidy-config" file would
  /// take precedence over ".clang-tidy" if both reside in the same directory.
  typedef std::vector<ConfigFileHandler> ConfigFileHandlers;

  FileOptionsBaseProvider(ClangTidyGlobalOptions GlobalOptions,
                          ClangTidyOptions DefaultOptions,
                          ClangTidyOptions OverrideOptions,
                          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS);

  FileOptionsBaseProvider(ClangTidyGlobalOptions GlobalOptions,
                          ClangTidyOptions DefaultOptions,
                          ClangTidyOptions OverrideOptions,
                          ConfigFileHandlers ConfigHandlers);

protected:
  void addRawFileOptions(llvm::StringRef AbsolutePath,
                         std::vector<OptionsSource> &CurOptions);

  /// Try to read configuration files from \p Directory using registered
  /// \c ConfigHandlers.
  llvm::Optional<OptionsSource> tryReadConfigFile(llvm::StringRef Directory);

  llvm::StringMap<OptionsSource> CachedOptions;
  ClangTidyOptions OverrideOptions;
  ConfigFileHandlers ConfigHandlers;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
};

/// Implementation of ClangTidyOptions interface, which is used for
/// '-config' command-line option.
class ConfigOptionsProvider : public FileOptionsBaseProvider {
public:
  ConfigOptionsProvider(
      ClangTidyGlobalOptions GlobalOptions, ClangTidyOptions DefaultOptions,
      ClangTidyOptions ConfigOptions, ClangTidyOptions OverrideOptions,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = nullptr);
  std::vector<OptionsSource> getRawOptions(llvm::StringRef FileName) override;

private:
  ClangTidyOptions ConfigOptions;
};

/// Implementation of the \c ClangTidyOptionsProvider interface, which
/// tries to find a configuration file in the closest parent directory of each
/// source file.
///
/// By default, files named ".clang-tidy" will be considered, and the
/// \c clang::tidy::parseConfiguration function will be used for parsing, but a
/// custom set of configuration file names and parsing functions can be
/// specified using the appropriate constructor.
class FileOptionsProvider : public FileOptionsBaseProvider {
public:
  /// Initializes the \c FileOptionsProvider instance.
  ///
  /// \param GlobalOptions are just stored and returned to the caller of
  /// \c getGlobalOptions.
  ///
  /// \param DefaultOptions are used for all settings not specified in a
  /// configuration file.
  ///
  /// If any of the \param OverrideOptions fields are set, they will override
  /// whatever options are read from the configuration file.
  FileOptionsProvider(
      ClangTidyGlobalOptions GlobalOptions, ClangTidyOptions DefaultOptions,
      ClangTidyOptions OverrideOptions,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = nullptr);

  /// Initializes the \c FileOptionsProvider instance with a custom set
  /// of configuration file handlers.
  ///
  /// \param GlobalOptions are just stored and returned to the caller of
  /// \c getGlobalOptions.
  ///
  /// \param DefaultOptions are used for all settings not specified in a
  /// configuration file.
  ///
  /// If any of the \param OverrideOptions fields are set, they will override
  /// whatever options are read from the configuration file.
  ///
  /// \param ConfigHandlers specifies a custom set of configuration file
  /// handlers. Each handler is a pair of configuration file name and a function
  /// that can parse configuration from this file type. The configuration files
  /// in each directory are searched for in the order of appearance in
  /// \p ConfigHandlers.
  FileOptionsProvider(ClangTidyGlobalOptions GlobalOptions,
                      ClangTidyOptions DefaultOptions,
                      ClangTidyOptions OverrideOptions,
                      ConfigFileHandlers ConfigHandlers);

  std::vector<OptionsSource> getRawOptions(llvm::StringRef FileName) override;
};

/// Parses LineFilter from JSON and stores it to the \p Options.
std::error_code parseLineFilter(llvm::StringRef LineFilter,
                                ClangTidyGlobalOptions &Options);

/// Parses configuration from JSON and returns \c ClangTidyOptions or an
/// error.
llvm::ErrorOr<ClangTidyOptions>
parseConfiguration(llvm::MemoryBufferRef Config);

using DiagCallback = llvm::function_ref<void(const llvm::SMDiagnostic &)>;

llvm::ErrorOr<ClangTidyOptions>
parseConfigurationWithDiags(llvm::MemoryBufferRef Config, DiagCallback Handler);

/// Serializes configuration to a YAML-encoded string.
std::string configurationAsText(const ClangTidyOptions &Options);

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYOPTIONS_H
