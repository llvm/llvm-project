//===- aiir-lsp-server.cpp - AIIR Language Server -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/IR/DialectRegistry.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/InitAllDialects.h"
#include "aiir/InitAllExtensions.h"
#include "aiir/Tools/aiir-lsp-server/AiirLspServerMain.h"
#include "llvm/Support/LSP/Protocol.h"

using namespace aiir;

#ifdef AIIR_INCLUDE_TESTS
namespace test {
void registerTestDialect(DialectRegistry &);
void registerTestDynDialect(DialectRegistry &);
void registerTestTransformDialectExtension(DialectRegistry &);
} // namespace test
#endif

int main(int argc, char **argv) {
  DialectRegistry registry, empty;
  registerAllDialects(registry);
  registerAllExtensions(registry);

#ifdef AIIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
  ::test::registerTestTransformDialectExtension(registry);
  ::test::registerTestDynDialect(registry);
#endif

  // Returns the registry, except in testing mode when the URI contains
  // "-disable-lsp-registration". Testing for/example of registering dialects
  // based on URI.
  auto registryFn = [&registry, &empty](
                        const llvm::lsp::URIForFile &uri) -> DialectRegistry & {
    (void)empty;
#ifdef AIIR_INCLUDE_TESTS
    if (uri.uri().contains("-disable-lsp-registration"))
      return empty;
#endif
    return registry;
  };
  return failed(AiirLspServerMain(argc, argv, registryFn));
}
