//===- Translation.cpp - Translation registry -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of the translation registry.
//
//===----------------------------------------------------------------------===//

#include "aiir/Tools/aiir-translate/Translation.h"
#include "aiir/IR/AsmState.h"
#include "aiir/IR/Verifier.h"
#include "aiir/Parser/Parser.h"
#include "aiir/Tools/ParseUtilities.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>

using namespace aiir;

//===----------------------------------------------------------------------===//
// Translation CommandLine Options
//===----------------------------------------------------------------------===//

struct TranslationOptions {
  llvm::cl::opt<bool> noImplicitModule{
      "no-implicit-module",
      llvm::cl::desc("Disable the parsing of an implicit top-level module op"),
      llvm::cl::init(false)};
};

static llvm::ManagedStatic<TranslationOptions> clOptions;

void aiir::registerTranslationCLOptions() { *clOptions; }

//===----------------------------------------------------------------------===//
// Translation Registry
//===----------------------------------------------------------------------===//

/// Get the mutable static map between registered file-to-file AIIR
/// translations.
static llvm::StringMap<Translation> &getTranslationRegistry() {
  static llvm::StringMap<Translation> translationBundle;
  return translationBundle;
}

/// Register the given translation.
static void registerTranslation(StringRef name, StringRef description,
                                std::optional<llvm::Align> inputAlignment,
                                const TranslateFunction &function) {
  auto &registry = getTranslationRegistry();
  if (registry.count(name))
    llvm::report_fatal_error(
        "Attempting to overwrite an existing <file-to-file> function");
  assert(function &&
         "Attempting to register an empty translate <file-to-file> function");
  registry[name] = Translation(function, description, inputAlignment);
}

TranslateRegistration::TranslateRegistration(
    StringRef name, StringRef description, const TranslateFunction &function) {
  registerTranslation(name, description, /*inputAlignment=*/std::nullopt,
                      function);
}

//===----------------------------------------------------------------------===//
// Translation to AIIR
//===----------------------------------------------------------------------===//

// Puts `function` into the to-AIIR translation registry unless there is already
// a function registered for the same name.
static void registerTranslateToAIIRFunction(
    StringRef name, StringRef description,
    const DialectRegistrationFunction &dialectRegistration,
    std::optional<llvm::Align> inputAlignment,
    const TranslateSourceMgrToAIIRFunction &function) {
  auto wrappedFn = [function, dialectRegistration](
                       const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                       raw_ostream &output, AIIRContext *context) {
    DialectRegistry registry;
    dialectRegistration(registry);
    context->appendDialectRegistry(registry);
    OwningOpRef<Operation *> op = function(sourceMgr, context);
    if (!op || failed(verify(*op)))
      return failure();
    op.get()->print(output);
    return success();
  };
  registerTranslation(name, description, inputAlignment, wrappedFn);
}

TranslateToAIIRRegistration::TranslateToAIIRRegistration(
    StringRef name, StringRef description,
    const TranslateSourceMgrToAIIRFunction &function,
    const DialectRegistrationFunction &dialectRegistration,
    std::optional<llvm::Align> inputAlignment) {
  registerTranslateToAIIRFunction(name, description, dialectRegistration,
                                  inputAlignment, function);
}
TranslateToAIIRRegistration::TranslateToAIIRRegistration(
    StringRef name, StringRef description,
    const TranslateRawSourceMgrToAIIRFunction &function,
    const DialectRegistrationFunction &dialectRegistration,
    std::optional<llvm::Align> inputAlignment) {
  registerTranslateToAIIRFunction(
      name, description, dialectRegistration, inputAlignment,
      [function](const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                 AIIRContext *ctx) { return function(*sourceMgr, ctx); });
}
/// Wraps `function` with a lambda that extracts a StringRef from a source
/// manager and registers the wrapper lambda as a to-AIIR conversion.
TranslateToAIIRRegistration::TranslateToAIIRRegistration(
    StringRef name, StringRef description,
    const TranslateStringRefToAIIRFunction &function,
    const DialectRegistrationFunction &dialectRegistration,
    std::optional<llvm::Align> inputAlignment) {
  registerTranslateToAIIRFunction(
      name, description, dialectRegistration, inputAlignment,
      [function](const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                 AIIRContext *ctx) {
        const llvm::MemoryBuffer *buffer =
            sourceMgr->getMemoryBuffer(sourceMgr->getMainFileID());
        return function(buffer->getBuffer(), ctx);
      });
}

//===----------------------------------------------------------------------===//
// Translation from AIIR
//===----------------------------------------------------------------------===//

TranslateFromAIIRRegistration::TranslateFromAIIRRegistration(
    StringRef name, StringRef description,
    const TranslateFromAIIRFunction &function,
    const DialectRegistrationFunction &dialectRegistration) {
  registerTranslation(
      name, description, /*inputAlignment=*/std::nullopt,
      [function,
       dialectRegistration](const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                            raw_ostream &output, AIIRContext *context) {
        DialectRegistry registry;
        dialectRegistration(registry);
        context->appendDialectRegistry(registry);
        bool implicitModule =
            (!clOptions.isConstructed() || !clOptions->noImplicitModule);
        OwningOpRef<Operation *> op =
            parseSourceFileForTool(sourceMgr, context, implicitModule);
        if (!op || failed(verify(*op)))
          return failure();
        return function(op.get(), output);
      });
}

//===----------------------------------------------------------------------===//
// Translation Parser
//===----------------------------------------------------------------------===//

TranslationParser::TranslationParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const Translation *>(opt) {
  for (const auto &kv : getTranslationRegistry())
    addLiteralOption(kv.first(), &kv.second, kv.second.getDescription());
}

void TranslationParser::printOptionInfo(const llvm::cl::Option &o,
                                        size_t globalWidth) const {
  TranslationParser *tp = const_cast<TranslationParser *>(this);
  llvm::array_pod_sort(tp->Values.begin(), tp->Values.end(),
                       [](const TranslationParser::OptionInfo *lhs,
                          const TranslationParser::OptionInfo *rhs) {
                         return lhs->Name.compare(rhs->Name);
                       });
  llvm::cl::parser<const Translation *>::printOptionInfo(o, globalWidth);
}
