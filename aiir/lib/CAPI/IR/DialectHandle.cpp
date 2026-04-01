//===- DialectHandle.cpp - C Interface for AIIR Dialect Operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/CAPI/Registration.h"

static inline const AiirDialectRegistrationHooks *
unwrap(AiirDialectHandle handle) {
  return (const AiirDialectRegistrationHooks *)handle.ptr;
}

AiirStringRef aiirDialectHandleGetNamespace(AiirDialectHandle handle) {
  return unwrap(handle)->getNamespaceHook();
}

void aiirDialectHandleInsertDialect(AiirDialectHandle handle,
                                    AiirDialectRegistry registry) {
  unwrap(handle)->insertHook(registry);
}

void aiirDialectHandleRegisterDialect(AiirDialectHandle handle,
                                      AiirContext ctx) {
  aiir::DialectRegistry registry;
  aiirDialectHandleInsertDialect(handle, wrap(&registry));
  unwrap(ctx)->appendDialectRegistry(registry);
}

AiirDialect aiirDialectHandleLoadDialect(AiirDialectHandle handle,
                                         AiirContext ctx) {
  return unwrap(handle)->loadHook(ctx);
}
