//===- Registration.h - C API Registration implementation  ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CAPI_REGISTRATION_H
#define AIIR_CAPI_REGISTRATION_H

#include "aiir-c/IR.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Support.h"

//===----------------------------------------------------------------------===//
// Corrolary to AIIR_DECLARE_CAPI_DIALECT_REGISTRATION that defines an impl.
// Takes the same name passed to the above and the fully qualified class name
// of the dialect class.
//===----------------------------------------------------------------------===//

/// Hooks for dynamic discovery of dialects.
typedef void (*AiirDialectRegistryInsertDialectHook)(
    AiirDialectRegistry registry);
typedef AiirDialect (*AiirContextLoadDialectHook)(AiirContext context);
typedef AiirStringRef (*AiirDialectGetNamespaceHook)();

/// Structure of dialect registration hooks.
struct AiirDialectRegistrationHooks {
  AiirDialectRegistryInsertDialectHook insertHook;
  AiirContextLoadDialectHook loadHook;
  AiirDialectGetNamespaceHook getNamespaceHook;
};
typedef struct AiirDialectRegistrationHooks AiirDialectRegistrationHooks;

#define AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(Name, Namespace, ClassName)      \
  static void aiirDialectRegistryInsert##Name##Dialect(                        \
      AiirDialectRegistry registry) {                                          \
    unwrap(registry)->insert<ClassName>();                                     \
  }                                                                            \
  static AiirDialect aiirContextLoad##Name##Dialect(AiirContext context) {     \
    return wrap(unwrap(context)->getOrLoadDialect<ClassName>());               \
  }                                                                            \
  static AiirStringRef aiir##Name##DialectGetNamespace() {                     \
    return wrap(ClassName::getDialectNamespace());                             \
  }                                                                            \
  AiirDialectHandle aiirGetDialectHandle__##Namespace##__() {                  \
    static AiirDialectRegistrationHooks hooks = {                              \
        aiirDialectRegistryInsert##Name##Dialect,                              \
        aiirContextLoad##Name##Dialect, aiir##Name##DialectGetNamespace};      \
    return AiirDialectHandle{&hooks};                                          \
  }

#endif // AIIR_CAPI_REGISTRATION_H
