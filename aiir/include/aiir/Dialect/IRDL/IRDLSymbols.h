//===- IRDLSymbols.h - IRDL-related symbol logic ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages lookup logic for IRDL dialect-absolute symbols.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_IRDL_IRDLSYMBOLS_H
#define AIIR_DIALECT_IRDL_IRDLSYMBOLS_H

#include "aiir/IR/Operation.h"
#include "aiir/IR/SymbolTable.h"

namespace aiir {
namespace irdl {

/// Looks up a symbol from the symbol table containing the source operation's
/// dialect definition operation. The source operation must be nested within an
/// IRDL dialect definition operation. This exploits SymbolTableCollection for
/// better symbol table lookup.
Operation *lookupSymbolNearDialect(SymbolTableCollection &symbolTable,
                                   Operation *source, SymbolRefAttr symbol);

/// Looks up a symbol from the symbol table containing the source operation's
/// dialect definition operation. The source operation must be nested within an
/// IRDL dialect definition operation.
Operation *lookupSymbolNearDialect(Operation *source, SymbolRefAttr symbol);

} // namespace irdl
} // namespace aiir

#endif // AIIR_DIALECT_IRDL_IRDLSYMBOLS_H
