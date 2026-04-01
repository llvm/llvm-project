//===- aiir-transform-opt.cpp -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/IR/Utils.h"
#include "aiir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "aiir/IR/AsmState.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/DialectRegistry.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/InitAllDialects.h"
#include "aiir/InitAllExtensions.h"
#include "aiir/InitAllPasses.h"
#include "aiir/Parser/Parser.h"
#include "aiir/Support/FileUtilities.h"
#include "aiir/Tools/aiir-opt/AiirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <cstdlib>

namespace {

using namespace llvm;

/// Structure containing command line options for the tool, these will get
/// initialized when an instance is created.
struct AiirTransformOptCLOptions {
  cl::opt<bool> allowUnregisteredDialects{
      "allow-unregistered-dialect",
      cl::desc("Allow operations coming from an unregistered dialect"),
      cl::init(false)};

  cl::opt<aiir::SourceMgrDiagnosticVerifierHandler::Level> verifyDiagnostics{
      "verify-diagnostics", llvm::cl::ValueOptional,
      cl::desc("Check that emitted diagnostics match expected-* lines on the "
               "corresponding line"),
      cl::values(
          clEnumValN(
              aiir::SourceMgrDiagnosticVerifierHandler::Level::All, "all",
              "Check all diagnostics (expected, unexpected, near-misses)"),
          // Implicit value: when passed with no arguments, e.g.
          // `--verify-diagnostics` or `--verify-diagnostics=`.
          clEnumValN(
              aiir::SourceMgrDiagnosticVerifierHandler::Level::All, "",
              "Check all diagnostics (expected, unexpected, near-misses)"),
          clEnumValN(
              aiir::SourceMgrDiagnosticVerifierHandler::Level::OnlyExpected,
              "only-expected", "Check only expected diagnostics"))};

  cl::opt<std::string> payloadFilename{cl::Positional, cl::desc("<input file>"),
                                       cl::init("-")};

  cl::opt<std::string> outputFilename{"o", cl::desc("Output filename"),
                                      cl::value_desc("filename"),
                                      cl::init("-")};

  cl::opt<std::string> transformMainFilename{
      "transform",
      cl::desc("File containing entry point of the transform script, if "
               "different from the input file"),
      cl::value_desc("filename"), cl::init("")};

  cl::list<std::string> transformLibraryFilenames{
      "transform-library", cl::desc("File(s) containing definitions of "
                                    "additional transform script symbols")};

  cl::opt<std::string> transformEntryPoint{
      "transform-entry-point",
      cl::desc("Name of the entry point transform symbol"),
      cl::init(aiir::transform::TransformDialect::kTransformEntryPointSymbolName
                   .str())};

  cl::opt<bool> disableExpensiveChecks{
      "disable-expensive-checks",
      cl::desc("Disables potentially expensive checks in the transform "
               "interpreter, providing more speed at the expense of "
               "potential memory problems and silent corruptions"),
      cl::init(false)};

  cl::opt<bool> dumpLibraryModule{
      "dump-library-module",
      cl::desc("Prints the combined library module before the output"),
      cl::init(false)};
};
} // namespace

/// "Managed" static instance of the command-line options structure. This makes
/// them locally-scoped and explicitly initialized/deinitialized. While this is
/// not strictly necessary in the tool source file that is not being used as a
/// library (where the options would pollute the global list of options), it is
/// good practice to follow this.
static llvm::ManagedStatic<AiirTransformOptCLOptions> clOptions;

/// Explicitly registers command-line options.
static void registerCLOptions() { *clOptions; }

namespace {
/// A wrapper class for source managers diagnostic. This provides both unique
/// ownership and virtual function-like overload for a pair of
/// inheritance-related classes that do not use virtual functions.
class DiagnosticHandlerWrapper {
public:
  /// Kind of the diagnostic handler to use.
  enum class Kind { EmitDiagnostics, VerifyDiagnostics };

  /// Constructs the diagnostic handler of the specified kind of the given
  /// source manager and context.
  DiagnosticHandlerWrapper(
      Kind kind, llvm::SourceMgr &mgr, aiir::AIIRContext *context,
      std::optional<aiir::SourceMgrDiagnosticVerifierHandler::Level> level =
          {}) {
    if (kind == Kind::EmitDiagnostics) {
      handler = new aiir::SourceMgrDiagnosticHandler(mgr, context);
    } else {
      assert(level.has_value() && "expected level");
      handler =
          new aiir::SourceMgrDiagnosticVerifierHandler(mgr, context, *level);
    }
  }

  /// This object is non-copyable but movable.
  DiagnosticHandlerWrapper(const DiagnosticHandlerWrapper &) = delete;
  DiagnosticHandlerWrapper(DiagnosticHandlerWrapper &&other) = default;
  DiagnosticHandlerWrapper &
  operator=(const DiagnosticHandlerWrapper &) = delete;
  DiagnosticHandlerWrapper &operator=(DiagnosticHandlerWrapper &&) = default;

  /// Verifies the captured "expected-*" diagnostics if required.
  llvm::LogicalResult verify() const {
    if (auto *ptr =
            dyn_cast<aiir::SourceMgrDiagnosticVerifierHandler *>(handler)) {
      return ptr->verify();
    }
    return aiir::success();
  }

  /// Destructs the object of the same type as allocated.
  ~DiagnosticHandlerWrapper() {
    if (auto *ptr = dyn_cast<aiir::SourceMgrDiagnosticHandler *>(handler)) {
      delete ptr;
    } else {
      delete cast<aiir::SourceMgrDiagnosticVerifierHandler *>(handler);
    }
  }

private:
  /// Internal storage is a type-safe union.
  llvm::PointerUnion<aiir::SourceMgrDiagnosticHandler *,
                     aiir::SourceMgrDiagnosticVerifierHandler *>
      handler;
};

/// AIIR has deeply rooted expectations that the LLVM source manager contains
/// exactly one buffer, until at least the lexer level. This class wraps
/// multiple LLVM source managers each managing a buffer to match AIIR's
/// expectations while still providing a centralized handling mechanism.
class TransformSourceMgr {
public:
  /// Constructs the source manager indicating whether diagnostic messages will
  /// be verified later on.
  explicit TransformSourceMgr(
      std::optional<aiir::SourceMgrDiagnosticVerifierHandler::Level>
          verifyDiagnostics)
      : verifyDiagnostics(verifyDiagnostics) {}

  /// Deconstructs the source manager. Note that `checkResults` must have been
  /// called on this instance before deconstructing it.
  ~TransformSourceMgr() {
    assert(resultChecked && "must check the result of diagnostic handlers by "
                            "running TransformSourceMgr::checkResult");
  }

  /// Parses the given buffer and creates the top-level operation of the kind
  /// specified as template argument in the given context. Additional parsing
  /// options may be provided.
  template <typename OpTy = aiir::Operation *>
  aiir::OwningOpRef<OpTy> parseBuffer(std::unique_ptr<MemoryBuffer> buffer,
                                      aiir::AIIRContext &context,
                                      const aiir::ParserConfig &config) {
    // Create a single-buffer LLVM source manager. Note that `unique_ptr` allows
    // the code below to capture a reference to the source manager in such a way
    // that it is not invalidated when the vector contents is eventually
    // reallocated.
    llvm::SourceMgr &mgr =
        *sourceMgrs.emplace_back(std::make_unique<llvm::SourceMgr>());
    mgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

    // Choose the type of diagnostic handler depending on whether diagnostic
    // verification needs to happen and store it.
    if (verifyDiagnostics) {
      diagHandlers.emplace_back(
          DiagnosticHandlerWrapper::Kind::VerifyDiagnostics, mgr, &context,
          verifyDiagnostics);
    } else {
      diagHandlers.emplace_back(DiagnosticHandlerWrapper::Kind::EmitDiagnostics,
                                mgr, &context);
    }

    // Defer to AIIR's parser.
    return aiir::parseSourceFile<OpTy>(mgr, config);
  }

  /// If diagnostic message verification has been requested upon construction of
  /// this source manager, performs the verification, reports errors and returns
  /// the result of the verification. Otherwise passes through the given value.
  llvm::LogicalResult checkResult(llvm::LogicalResult result) {
    resultChecked = true;
    if (!verifyDiagnostics)
      return result;

    return aiir::failure(llvm::any_of(diagHandlers, [](const auto &handler) {
      return aiir::failed(handler.verify());
    }));
  }

private:
  /// Indicates whether diagnostic message verification is requested.
  const std::optional<aiir::SourceMgrDiagnosticVerifierHandler::Level>
      verifyDiagnostics;

  /// Indicates that diagnostic message verification has taken place, and the
  /// deconstruction is therefore safe.
  bool resultChecked = false;

  /// Storage for per-buffer source managers and diagnostic handlers. These are
  /// wrapped into unique pointers in order to make it safe to capture
  /// references to these objects: if the vector is reallocated, the unique
  /// pointer objects are moved by the pointer addresses won't change. Also, for
  /// handlers, this allows to store the pointer to the base class.
  SmallVector<std::unique_ptr<llvm::SourceMgr>> sourceMgrs;
  SmallVector<DiagnosticHandlerWrapper> diagHandlers;
};
} // namespace

/// Trivial wrapper around `applyTransforms` that doesn't support extra mapping
/// and doesn't enforce the entry point transform ops being top-level.
static llvm::LogicalResult
applyTransforms(aiir::Operation *payloadRoot,
                aiir::transform::TransformOpInterface transformRoot,
                const aiir::transform::TransformOptions &options) {
  return applyTransforms(payloadRoot, transformRoot, {}, options,
                         /*enforceToplevelTransformOp=*/false);
}

/// Applies transforms indicated in the transform dialect script to the input
/// buffer. The transform script may be embedded in the input buffer or as a
/// separate buffer. The transform script may have external symbols, the
/// definitions of which must be provided in transform library buffers. If the
/// application is successful, prints the transformed input buffer into the
/// given output stream. Additional configuration options are derived from
/// command-line options.
static llvm::LogicalResult processPayloadBuffer(
    raw_ostream &os, std::unique_ptr<MemoryBuffer> inputBuffer,
    std::unique_ptr<llvm::MemoryBuffer> transformBuffer,
    MutableArrayRef<std::unique_ptr<MemoryBuffer>> transformLibraries,
    aiir::DialectRegistry &registry) {

  // Initialize the AIIR context, and various configurations.
  aiir::AIIRContext context(registry, aiir::AIIRContext::Threading::DISABLED);
  context.allowUnregisteredDialects(clOptions->allowUnregisteredDialects);
  aiir::ParserConfig config(&context);
  TransformSourceMgr sourceMgr(
      /*verifyDiagnostics=*/clOptions->verifyDiagnostics.getNumOccurrences()
          ? std::optional{clOptions->verifyDiagnostics.getValue()}
          : std::nullopt);

  // Parse the input buffer that will be used as transform payload.
  aiir::OwningOpRef<aiir::Operation *> payloadRoot =
      sourceMgr.parseBuffer(std::move(inputBuffer), context, config);
  if (!payloadRoot)
    return sourceMgr.checkResult(aiir::failure());

  // Identify the module containing the transform script entry point. This may
  // be the same module as the input or a separate module. In the former case,
  // make a copy of the module so it can be modified freely. Modification may
  // happen in the script itself (at which point it could be rewriting itself
  // during interpretation, leading to tricky memory errors) or by embedding
  // library modules in the script.
  aiir::OwningOpRef<aiir::ModuleOp> transformRoot;
  if (transformBuffer) {
    transformRoot = sourceMgr.parseBuffer<aiir::ModuleOp>(
        std::move(transformBuffer), context, config);
    if (!transformRoot)
      return sourceMgr.checkResult(aiir::failure());
  } else {
    transformRoot = cast<aiir::ModuleOp>(payloadRoot->clone());
  }

  // Parse and merge the libraries into the main transform module.
  for (auto &&transformLibrary : transformLibraries) {
    aiir::OwningOpRef<aiir::ModuleOp> libraryModule =
        sourceMgr.parseBuffer<aiir::ModuleOp>(std::move(transformLibrary),
                                              context, config);

    if (!libraryModule ||
        aiir::failed(aiir::transform::detail::mergeSymbolsInto(
            *transformRoot, std::move(libraryModule))))
      return sourceMgr.checkResult(aiir::failure());
  }

  // If requested, dump the combined transform module.
  if (clOptions->dumpLibraryModule)
    transformRoot->dump();

  // Find the entry point symbol. Even if it had originally been in the payload
  // module, it was cloned into the transform module so only look there.
  aiir::transform::TransformOpInterface entryPoint =
      aiir::transform::detail::findTransformEntryPoint(
          *transformRoot, aiir::ModuleOp(), clOptions->transformEntryPoint);
  if (!entryPoint)
    return sourceMgr.checkResult(aiir::failure());

  // Apply the requested transformations.
  aiir::transform::TransformOptions transformOptions;
  transformOptions.enableExpensiveChecks(!clOptions->disableExpensiveChecks);
  if (aiir::failed(applyTransforms(*payloadRoot, entryPoint, transformOptions)))
    return sourceMgr.checkResult(aiir::failure());

  // Print the transformed result and check the captured diagnostics if
  // requested.
  payloadRoot->print(os);
  return sourceMgr.checkResult(aiir::success());
}

/// Tool entry point.
static llvm::LogicalResult runMain(int argc, char **argv) {
  // Register all upstream dialects and extensions. Specific uses are advised
  // not to register all dialects indiscriminately but rather hand-pick what is
  // necessary for their use case.
  aiir::DialectRegistry registry;
  aiir::registerAllDialects(registry);
  aiir::registerAllExtensions(registry);
  aiir::registerAllPasses();

  // Explicitly register the transform dialect. This is not strictly necessary
  // since it has been already registered as part of the upstream dialect list,
  // but useful for example purposes for cases when dialects to register are
  // hand-picked. The transform dialect must be registered.
  registry.insert<aiir::transform::TransformDialect>();

  // Register various command-line options. Note that the LLVM initializer
  // object is a RAII that ensures correct deconstruction of command-line option
  // objects inside ManagedStatic.
  llvm::InitLLVM y(argc, argv);
  aiir::registerAsmPrinterCLOptions();
  aiir::registerAIIRContextCLOptions();
  registerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Minimal Transform dialect driver\n");

  // Try opening the main input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> payloadFile =
      aiir::openInputFile(clOptions->payloadFilename, &errorMessage);
  if (!payloadFile) {
    llvm::errs() << errorMessage << "\n";
    return aiir::failure();
  }

  // Try opening the output file.
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      aiir::openOutputFile(clOptions->outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return aiir::failure();
  }

  // Try opening the main transform file if provided.
  std::unique_ptr<llvm::MemoryBuffer> transformRootFile;
  if (!clOptions->transformMainFilename.empty()) {
    if (clOptions->transformMainFilename == clOptions->payloadFilename) {
      llvm::errs() << "warning: " << clOptions->payloadFilename
                   << " is provided as both payload and transform file\n";
    } else {
      transformRootFile =
          aiir::openInputFile(clOptions->transformMainFilename, &errorMessage);
      if (!transformRootFile) {
        llvm::errs() << errorMessage << "\n";
        return aiir::failure();
      }
    }
  }

  // Try opening transform library files if provided.
  SmallVector<std::unique_ptr<llvm::MemoryBuffer>> transformLibraries;
  transformLibraries.reserve(clOptions->transformLibraryFilenames.size());
  for (llvm::StringRef filename : clOptions->transformLibraryFilenames) {
    transformLibraries.emplace_back(
        aiir::openInputFile(filename, &errorMessage));
    if (!transformLibraries.back()) {
      llvm::errs() << errorMessage << "\n";
      return aiir::failure();
    }
  }

  return processPayloadBuffer(outputFile->os(), std::move(payloadFile),
                              std::move(transformRootFile), transformLibraries,
                              registry);
}

int main(int argc, char **argv) {
  return aiir::asMainReturnCode(runMain(argc, argv));
}
