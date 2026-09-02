//===--- FlangTidyOptions.h - flang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYOPTIONS_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYOPTIONS_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Fortran::tidy {

/// Global options for FlangTidy. These options are neither stored nor read from
/// configuration files.
struct FlangTidyGlobalOptions {
  // Reserved for future use (line filters, etc.)
};

/// Contains options for flang-tidy. These options may be read from
/// configuration files, and may be different for different translation units.
struct FlangTidyOptions {
  /// These options are used for all settings that haven't been
  /// overridden by the \c OptionsProvider.
  static FlangTidyOptions getDefaults();

  /// Overwrites all fields in here by the fields of \p Other that have a value.
  /// \p Order specifies precedence of \p Other option.
  FlangTidyOptions &mergeWith(const FlangTidyOptions &Other, unsigned Order);

  /// Creates a new \c FlangTidyOptions instance combined from all fields
  /// of this instance overridden by the fields of \p Other that have a value.
  /// \p Order specifies precedence of \p Other option.
  [[nodiscard]] FlangTidyOptions merge(const FlangTidyOptions &Other,
                                       unsigned Order) const;

  /// Checks filter.
  std::optional<std::string> Checks;

  /// WarningsAsErrors filter.
  std::optional<std::string> WarningsAsErrors;

  /// Helper structure for storing option value with priority of the value.
  struct FlangTidyValue {
    FlangTidyValue() = default;
    FlangTidyValue(const char *Value) : Value(Value) {}
    FlangTidyValue(llvm::StringRef Value, unsigned Priority = 0)
        : Value(Value), Priority(Priority) {}

    std::string Value;
    /// Priority stores relative precedence of the value loaded from config
    /// files to disambiguate local vs global value from different levels.
    unsigned Priority = 0;
  };
  using StringPair = std::pair<std::string, std::string>;
  using OptionMap = llvm::StringMap<FlangTidyValue>;

  /// Key-value mapping used to store check-specific options.
  OptionMap CheckOptions;

  using ArgList = std::vector<std::string>;

  /// Add extra compilation arguments to the end of the list.
  std::optional<ArgList> ExtraArgs;

  /// Add extra compilation arguments to the start of the list.
  std::optional<ArgList> ExtraArgsBefore;

  /// Only used in the FileOptionsProvider and ConfigOptionsProvider. If true
  /// and using a FileOptionsProvider, it will take a configuration file in the
  /// parent directory (if any exists) and apply this config file on top of the
  /// parent one. If false or missing, only this configuration file will be
  /// used.
  std::optional<bool> InheritParentConfig;

  // Runtime-only options (not serialized to/from YAML)
  std::vector<std::string> sourcePaths;   // Set by command line
  std::vector<std::string> enabledChecks; // Parsed from Checks string
  std::vector<std::string>
      enabledWarningsAsErrors; // Parsed from WarningsAsErrors string
  const char *argv0 = nullptr; // Set by command line

  /// Parse the Checks string into enabledChecks vector
  void parseChecksString();

  /// Parse the WarningsAsErrors string into enabledWarningsAsErrors vector
  void parseWarningsAsErrorsString();
};

/// Abstract interface for retrieving various FlangTidy options.
class FlangTidyOptionsProvider {
public:
  static const char OptionsSourceTypeDefaultBinary[];
  static const char OptionsSourceTypeCheckCommandLineOption[];
  static const char OptionsSourceTypeConfigCommandLineOption[];

  virtual ~FlangTidyOptionsProvider() {}

  /// Returns global options, which are independent of the file.
  virtual const FlangTidyGlobalOptions &getGlobalOptions() = 0;

  /// FlangTidyOptions and its source.
  using OptionsSource = std::pair<FlangTidyOptions, std::string>;

  /// Returns an ordered vector of OptionsSources, in order of increasing
  /// priority.
  virtual std::vector<OptionsSource>
  getRawOptions(llvm::StringRef FileName) = 0;

  /// Returns options applying to a specific translation unit with the
  /// specified \p FileName.
  FlangTidyOptions getOptions(llvm::StringRef FileName);
};

/// Implementation of the \c FlangTidyOptionsProvider interface, which
/// returns the same options for all files.
class DefaultOptionsProvider : public FlangTidyOptionsProvider {
public:
  DefaultOptionsProvider(FlangTidyGlobalOptions GlobalOptions,
                         FlangTidyOptions Options)
      : GlobalOptions(std::move(GlobalOptions)),
        DefaultOptions(std::move(Options)) {}
  const FlangTidyGlobalOptions &getGlobalOptions() override {
    return GlobalOptions;
  }
  std::vector<OptionsSource> getRawOptions(llvm::StringRef FileName) override;

protected:
  FlangTidyGlobalOptions GlobalOptions;
  FlangTidyOptions DefaultOptions;
};

class FileOptionsBaseProvider : public DefaultOptionsProvider {
protected:
  // A pair of configuration file base name and a function parsing
  // configuration from text in the corresponding format.
  using ConfigFileHandler =
      std::pair<std::string, std::function<llvm::ErrorOr<FlangTidyOptions>(
                                 llvm::MemoryBufferRef)>>;

  /// Configuration file handlers listed in the order of priority.
  using ConfigFileHandlers = std::vector<ConfigFileHandler>;

  FileOptionsBaseProvider(FlangTidyGlobalOptions GlobalOptions,
                          FlangTidyOptions DefaultOptions,
                          FlangTidyOptions OverrideOptions,
                          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS);

  FileOptionsBaseProvider(FlangTidyGlobalOptions GlobalOptions,
                          FlangTidyOptions DefaultOptions,
                          FlangTidyOptions OverrideOptions,
                          ConfigFileHandlers ConfigHandlers);

  void addRawFileOptions(llvm::StringRef AbsolutePath,
                         std::vector<OptionsSource> &CurOptions);

  llvm::ErrorOr<llvm::SmallString<128>>
  getNormalizedAbsolutePath(llvm::StringRef AbsolutePath);

  /// Try to read configuration files from \p Directory using registered
  /// \c ConfigHandlers.
  std::optional<OptionsSource> tryReadConfigFile(llvm::StringRef Directory);

  struct OptionsCache {
    llvm::StringMap<size_t> Memorized;
    llvm::SmallVector<OptionsSource, 4U> Storage;
  } CachedOptions;
  FlangTidyOptions OverrideOptions;
  ConfigFileHandlers ConfigHandlers;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
};

/// Implementation of FlangTidyOptions interface, which is used for
/// '-config' command-line option.
class ConfigOptionsProvider : public FileOptionsBaseProvider {
public:
  ConfigOptionsProvider(
      FlangTidyGlobalOptions GlobalOptions, FlangTidyOptions DefaultOptions,
      FlangTidyOptions ConfigOptions, FlangTidyOptions OverrideOptions,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = nullptr);
  std::vector<OptionsSource> getRawOptions(llvm::StringRef FileName) override;

private:
  FlangTidyOptions ConfigOptions;
};

/// Implementation of the \c FlangTidyOptionsProvider interface, which
/// tries to find a configuration file in the closest parent directory of each
/// source file.
class FileOptionsProvider : public FileOptionsBaseProvider {
public:
  FileOptionsProvider(
      FlangTidyGlobalOptions GlobalOptions, FlangTidyOptions DefaultOptions,
      FlangTidyOptions OverrideOptions,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = nullptr);

  FileOptionsProvider(FlangTidyGlobalOptions GlobalOptions,
                      FlangTidyOptions DefaultOptions,
                      FlangTidyOptions OverrideOptions,
                      ConfigFileHandlers ConfigHandlers);

  std::vector<OptionsSource> getRawOptions(llvm::StringRef FileName) override;
};

/// Parse FlangTidy configuration from YAML and returns \c FlangTidyOptions or
/// an error.
llvm::ErrorOr<FlangTidyOptions>
parseConfiguration(llvm::MemoryBufferRef Config);

/// Serializes configuration to a YAML-encoded string.
std::string configurationAsText(const FlangTidyOptions &Options);

/// Simple function to get options for a file (for backward compatibility)
FlangTidyOptions getOptionsForFile(llvm::StringRef FileName);

} // namespace Fortran::tidy

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYOPTIONS_H
