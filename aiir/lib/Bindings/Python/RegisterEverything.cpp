//===- RegisterEverything.cpp - API to register all dialects/passes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/RegisterEverything.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

NB_MODULE(_aiirRegisterEverything, m) {
  m.doc() = "AIIR All Upstream Dialects, Translations and Passes Registration";

  m.def("register_dialects", [](AiirDialectRegistry registry) {
    aiirRegisterAllDialects(registry);
  });
  m.def("register_llvm_translations",
        [](AiirContext context) { aiirRegisterAllLLVMTranslations(context); });

  // Register all passes on load.
  aiirRegisterAllPasses();
}
