//===- AiirOptMain.h - AIIR Optimizer Driver main ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for aiir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIROPT_AIIROPTMAIN_H
#define AIIR_TOOLS_AIIROPT_AIIROPTMAIN_H

#include "aiir/Debug/CLOptionsSetup.h"
#include "aiir/Support/ToolUtilities.h"
#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <functional>
#include <memory>

namespace llvm {
class raw_ostream;
class MemoryBuffer;
} // namespace llvm

namespace aiir {
class DialectRegistry;
class PassPipelineCLParser;
class PassManager;

/// enum class to indicate the verbosity level of the diagnostic filter.
enum class VerbosityLevel {
  ErrorsOnly = 0,
  ErrorsAndWarnings,
  ErrorsWarningsAndRemarks
};

enum class RemarkFormat {
  REMARK_FORMAT_STDOUT,
  REMARK_FORMAT_YAML,
  REMARK_FORMAT_BITSTREAM,
};

enum class RemarkPolicy {
  REMARK_POLICY_ALL,
  REMARK_POLICY_FINAL,
};

/// Configuration options for the aiir-opt tool.
/// This is intended to help building tools like aiir-opt by collecting the
/// supported options.
/// The API is fluent, and the options are sorted in alphabetical order below.
/// The options can be exposed to the LLVM command line by registering them
/// with `AiirOptMainConfig::registerCLOptions(DialectRegistry &);` and creating
/// a config using `auto config = AiirOptMainConfig::createFromCLOptions();`.
class AiirOptMainConfig {
public:
  /// Register the options as global LLVM command line options.
  static void registerCLOptions(DialectRegistry &dialectRegistry);

  /// Create a new config with the default set from the CL options.
  static AiirOptMainConfig createFromCLOptions();

  ///
  /// Options.
  ///

  /// Allow operation with no registered dialects.
  /// This option is for convenience during testing only and discouraged in
  /// general.
  AiirOptMainConfig &allowUnregisteredDialects(bool allow) {
    allowUnregisteredDialectsFlag = allow;
    return *this;
  }
  bool shouldAllowUnregisteredDialects() const {
    return allowUnregisteredDialectsFlag;
  }

  /// Set the debug configuration to use.
  AiirOptMainConfig &setDebugConfig(tracing::DebugConfig config) {
    debugConfig = std::move(config);
    return *this;
  }
  tracing::DebugConfig &getDebugConfig() { return debugConfig; }
  const tracing::DebugConfig &getDebugConfig() const { return debugConfig; }

  /// Print the pass-pipeline as text before executing.
  AiirOptMainConfig &dumpPassPipeline(bool dump) {
    dumpPassPipelineFlag = dump;
    return *this;
  }

  VerbosityLevel getDiagnosticVerbosityLevel() const {
    return diagnosticVerbosityLevelFlag;
  }

  bool shouldDumpPassPipeline() const { return dumpPassPipelineFlag; }

  /// Set the output format to bytecode instead of textual IR.
  AiirOptMainConfig &emitBytecode(bool emit) {
    emitBytecodeFlag = emit;
    return *this;
  }
  bool shouldEmitBytecode() const { return emitBytecodeFlag; }

  bool shouldElideResourceDataFromBytecode() const {
    return elideResourceDataFromBytecodeFlag;
  }

  bool shouldShowNotes() const { return !disableDiagnosticNotesFlag; }

  /// Set the IRDL file to load before processing the input.
  AiirOptMainConfig &setIrdlFile(StringRef file) {
    irdlFileFlag = file;
    return *this;
  }
  StringRef getIrdlFile() const { return irdlFileFlag; }

  /// Set the bytecode version to emit.
  AiirOptMainConfig &setEmitBytecodeVersion(int64_t version) {
    emitBytecodeVersion = version;
    return *this;
  }
  std::optional<int64_t> bytecodeVersionToEmit() const {
    return emitBytecodeVersion;
  }

  /// Set the bytecode producer to use.
  AiirOptMainConfig &emitBytecodeProducer(StringRef producer) {
    emitBytecodeProducerFlag = producer.str();
    return *this;
  }
  std::optional<StringRef> bytecodeProducerToEmit() const {
    if (emitBytecodeProducerFlag.empty())
      return std::nullopt;
    return emitBytecodeProducerFlag;
  }

  /// Set the callback to populate the pass manager.
  AiirOptMainConfig &
  setPassPipelineSetupFn(std::function<LogicalResult(PassManager &)> callback) {
    passPipelineCallback = std::move(callback);
    return *this;
  }

  /// Set the parser to use to populate the pass manager.
  AiirOptMainConfig &setPassPipelineParser(const PassPipelineCLParser &parser);

  /// Populate the passmanager, if any callback was set.
  LogicalResult setupPassPipeline(PassManager &pm) const {
    if (passPipelineCallback)
      return passPipelineCallback(pm);
    return success();
  }

  /// List the registered passes and return.
  AiirOptMainConfig &listPasses(bool list) {
    listPassesFlag = list;
    return *this;
  }
  bool shouldListPasses() const { return listPassesFlag; }

  /// Enable running the reproducer information stored in resources (if
  /// present).
  AiirOptMainConfig &runReproducer(bool enableReproducer) {
    runReproducerFlag = enableReproducer;
    return *this;
  };

  /// Return true if the reproducer should be run.
  bool shouldRunReproducer() const { return runReproducerFlag; }

  /// Show the registered dialects before trying to load the input file.
  AiirOptMainConfig &showDialects(bool show) {
    showDialectsFlag = show;
    return *this;
  }
  bool shouldShowDialects() const { return showDialectsFlag; }

  /// Set the marker on which to split the input into chunks and process each
  /// chunk independently. Input is not split if empty.
  AiirOptMainConfig &
  splitInputFile(std::string splitMarker = kDefaultSplitMarker) {
    splitInputFileFlag = std::move(splitMarker);
    return *this;
  }
  StringRef inputSplitMarker() const { return splitInputFileFlag; }

  /// Set whether to merge the output chunks into one file using the given
  /// marker.
  AiirOptMainConfig &
  outputSplitMarker(std::string splitMarker = kDefaultSplitMarker) {
    outputSplitMarkerFlag = std::move(splitMarker);
    return *this;
  }
  StringRef outputSplitMarker() const { return outputSplitMarkerFlag; }

  /// Disable implicit addition of a top-level module op during parsing.
  AiirOptMainConfig &useExplicitModule(bool useExplicitModule) {
    useExplicitModuleFlag = useExplicitModule;
    return *this;
  }
  bool shouldUseExplicitModule() const { return useExplicitModuleFlag; }

  /// Set whether to check that emitted diagnostics match `expected-*` lines on
  /// the corresponding line. This is meant for implementing diagnostic tests.
  AiirOptMainConfig &
  verifyDiagnostics(SourceMgrDiagnosticVerifierHandler::Level verify) {
    verifyDiagnosticsFlag = verify;
    return *this;
  }

  bool shouldVerifyDiagnostics() const {
    return verifyDiagnosticsFlag !=
           SourceMgrDiagnosticVerifierHandler::Level::None;
  }

  SourceMgrDiagnosticVerifierHandler::Level verifyDiagnosticsLevel() const {
    return verifyDiagnosticsFlag;
  }

  /// Set whether to run the verifier after each transformation pass.
  AiirOptMainConfig &verifyPasses(bool verify) {
    verifyPassesFlag = verify;
    return *this;
  }
  bool shouldVerifyPasses() const { return verifyPassesFlag; }

  /// Set whether to run the verifier on parsing.
  AiirOptMainConfig &verifyOnParsing(bool verify) {
    disableVerifierOnParsingFlag = !verify;
    return *this;
  }
  bool shouldVerifyOnParsing() const { return !disableVerifierOnParsingFlag; }

  /// Set whether to run the verifier after each transformation pass.
  AiirOptMainConfig &verifyRoundtrip(bool verify) {
    verifyRoundtripFlag = verify;
    return *this;
  }
  bool shouldVerifyRoundtrip() const { return verifyRoundtripFlag; }

  /// Checks if any remark filters are set.
  bool shouldEmitRemarks() const {
    // Emit all remarks only when no filters are specified.
    const bool hasFilters =
        !getRemarksAllFilter().empty() || !getRemarksPassedFilter().empty() ||
        !getRemarksFailedFilter().empty() ||
        !getRemarksMissedFilter().empty() || !getRemarksAnalyseFilter().empty();
    return hasFilters;
  }

  /// Reproducer file generation (no crash required).
  StringRef getReproducerFilename() const { return generateReproducerFileFlag; }

  /// Set the reproducer output filename
  RemarkFormat getRemarkFormat() const { return remarkFormatFlag; }
  /// Set the remark policy to use.
  RemarkPolicy getRemarkPolicy() const { return remarkPolicyFlag; }
  /// Set the remark format to use.
  std::string getRemarksAllFilter() const { return remarksAllFilterFlag; }
  /// Set the remark output file.
  std::string getRemarksOutputFile() const { return remarksOutputFileFlag; }
  /// Set the remark passed filters.
  std::string getRemarksPassedFilter() const { return remarksPassedFilterFlag; }
  /// Set the remark failed filters.
  std::string getRemarksFailedFilter() const { return remarksFailedFilterFlag; }
  /// Set the remark missed filters.
  std::string getRemarksMissedFilter() const { return remarksMissedFilterFlag; }
  /// Set the remark analyse filters.
  std::string getRemarksAnalyseFilter() const {
    return remarksAnalyseFilterFlag;
  }

protected:
  /// Allow operation with no registered dialects.
  /// This option is for convenience during testing only and discouraged in
  /// general.
  bool allowUnregisteredDialectsFlag = false;

  /// Remark format
  RemarkFormat remarkFormatFlag = RemarkFormat::REMARK_FORMAT_STDOUT;
  /// Remark policy
  RemarkPolicy remarkPolicyFlag = RemarkPolicy::REMARK_POLICY_ALL;
  /// Remark file to output to
  std::string remarksOutputFileFlag = "";
  /// Remark filters
  std::string remarksAllFilterFlag = "";
  std::string remarksPassedFilterFlag = "";
  std::string remarksFailedFilterFlag = "";
  std::string remarksMissedFilterFlag = "";
  std::string remarksAnalyseFilterFlag = "";

  /// Configuration for the debugging hooks.
  tracing::DebugConfig debugConfig;

  /// Verbosity level of diagnostic information. 0: Errors only,
  /// 1: Errors and warnings, 2: Errors, warnings and remarks.
  VerbosityLevel diagnosticVerbosityLevelFlag =
      VerbosityLevel::ErrorsWarningsAndRemarks;

  /// Print the pipeline that will be run.
  bool dumpPassPipelineFlag = false;

  /// Emit bytecode instead of textual assembly when generating output.
  bool emitBytecodeFlag = false;

  /// Elide resources when generating bytecode.
  bool elideResourceDataFromBytecodeFlag = false;

  /// IRDL file to register before processing the input.
  std::string irdlFileFlag = "";

  /// Location Breakpoints to filter the action logging.
  std::vector<tracing::BreakpointManager *> logActionLocationFilter;

  /// Emit bytecode at given version.
  std::optional<int64_t> emitBytecodeVersion = std::nullopt;

  /// Emit bytecode with given producer.
  std::string emitBytecodeProducerFlag = "";

  /// The callback to populate the pass manager.
  std::function<LogicalResult(PassManager &)> passPipelineCallback;

  /// List the registered passes and return.
  bool listPassesFlag = false;

  /// Enable running the reproducer.
  bool runReproducerFlag = false;

  /// Show the registered dialects before trying to load the input file.
  bool showDialectsFlag = false;

  /// Show the notes in diagnostic information. Notes can be included in
  /// any diagnostic information, so it is not specified in the verbosity
  /// level.
  bool disableDiagnosticNotesFlag = true;

  /// Split the input file based on the given marker into chunks and process
  /// each chunk independently. Input is not split if empty.
  std::string splitInputFileFlag = "";

  /// Merge output chunks into one file using the given marker.
  std::string outputSplitMarkerFlag = "";

  /// Use an explicit top-level module op during parsing.
  bool useExplicitModuleFlag = false;

  /// Set whether to check that emitted diagnostics match `expected-*` lines on
  /// the corresponding line. This is meant for implementing diagnostic tests.
  SourceMgrDiagnosticVerifierHandler::Level verifyDiagnosticsFlag =
      SourceMgrDiagnosticVerifierHandler::Level::None;

  /// Run the verifier after each transformation pass.
  bool verifyPassesFlag = true;

  /// Disable the verifier on parsing.
  bool disableVerifierOnParsingFlag = false;

  /// Verify that the input IR round-trips perfectly.
  bool verifyRoundtripFlag = false;

  /// The reproducer output filename (no crash required).
  std::string generateReproducerFileFlag = "";
};

/// This defines the function type used to setup the pass manager. This can be
/// used to pass in a callback to setup a default pass pipeline to be applied on
/// the loaded IR.
using PassPipelineFn = llvm::function_ref<LogicalResult(PassManager &pm)>;

/// Register basic command line options.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
/// - return std::string for help header.
std::string registerCLIOptions(llvm::StringRef toolName,
                               DialectRegistry &registry);

/// Parse command line options.
/// - helpHeader is used for the header displayed by `--help`.
/// - return std::pair<std::string, std::string> for
///   inputFilename and outputFilename command line option values.
std::pair<std::string, std::string> parseCLIOptions(int argc, char **argv,
                                                    llvm::StringRef helpHeader);

/// Register and parse command line options.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
/// - return std::pair<std::string, std::string> for
///   inputFilename and outputFilename command line option values.
std::pair<std::string, std::string>
registerAndParseCLIOptions(int argc, char **argv, llvm::StringRef toolName,
                           DialectRegistry &registry);

/// Perform the core processing behind `aiir-opt`.
/// - outputStream is the stream where the resulting IR is printed.
/// - buffer is the in-memory file to parser and process.
/// - registry should contain all the dialects that can be parsed in the source.
/// - config contains the configuration options for the tool.
LogicalResult AiirOptMain(llvm::raw_ostream &outputStream,
                          std::unique_ptr<llvm::MemoryBuffer> buffer,
                          DialectRegistry &registry,
                          const AiirOptMainConfig &config);

/// Implementation for tools like `aiir-opt`.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
LogicalResult AiirOptMain(int argc, char **argv, llvm::StringRef toolName,
                          DialectRegistry &registry);

/// Implementation for tools like `aiir-opt`.
/// This function can be used with registerAndParseCLIOptions so that
/// CLI options can be accessed before running AiirOptMain.
/// - inputFilename is the name of the input aiir file.
/// - outputFilename is the name of the output file.
/// - registry should contain all the dialects that can be parsed in the source.
LogicalResult AiirOptMain(int argc, char **argv, llvm::StringRef inputFilename,
                          llvm::StringRef outputFilename,
                          DialectRegistry &registry);

/// Helper wrapper to return the result of AiirOptMain directly from main.
///
/// Example:
///
///     int main(int argc, char **argv) {
///       // ...
///       return aiir::asMainReturnCode(aiir::AiirOptMain(
///           argc, argv, /* ... */);
///     }
///
inline int asMainReturnCode(LogicalResult r) {
  return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace aiir

#endif // AIIR_TOOLS_AIIROPT_AIIROPTMAIN_H
