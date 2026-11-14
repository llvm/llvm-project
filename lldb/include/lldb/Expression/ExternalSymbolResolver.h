//===-- ExternalSymbolResolver.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_EXPRESSION_EXTERNALSYMBOLRESOLVER_H
#define LLDB_EXPRESSION_EXTERNALSYMBOLRESOLVER_H

#include <vector>

#include "lldb/Core/ModuleList.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

/// \class ExternalSymbolResolver ExternalSymbolResolver.h
/// "lldb/Expression/ExternalSymbolResolver.h" Finds the load addresses of
/// symbols that an expression refers to but does not define.
///
/// An expression's IR refers by name to functions and variables that live
/// outside the expression's own module. Finding the address for such a name is
/// not a simple lookup: the name may be an LLDB function call label, it may
/// need alternate manglings tried, and it may be satisfied by the target's
/// images, by a language runtime, or by a symbol that a previous expression
/// defined (see Target::GetPersistentSymbol).
class ExternalSymbolResolver {
public:
  /// Constructor
  ///
  /// \param[in] sym_ctx
  ///     The symbol context to search from. Its module, if any, is searched
  ///     before all others.
  ///
  /// \param[in] strip_underscore
  ///     True for platforms where global symbols have a _ prefix, in which case
  ///     the name with that prefix removed is also tried.
  ExternalSymbolResolver(const SymbolContext &sym_ctx, bool strip_underscore);

  /// Search the modules of \p contexts before any others, except for the
  /// module of the symbol context this resolver was constructed with.
  void AppendPreferredSymbolContexts(SymbolContextList const &contexts) {
    for (auto const &ctx : contexts)
      if (ctx.module_sp)
        m_preferred_modules.Append(ctx.module_sp);
  }

  /// Find the load address of \p name.
  ///
  /// \param[out] missing_weak
  ///     Set to true if \p name was only found as a weak undefined symbol, in
  ///     which case the returned address is 0.
  ///
  /// \return
  ///     The load address, or LLDB_INVALID_ADDRESS if \p name was not found.
  lldb::addr_t FindSymbol(ConstString name, bool &missing_weak);

private:
  void CollectCandidateCNames(std::vector<ConstString> &C_names,
                              ConstString name);

  void CollectCandidateCPlusPlusNames(std::vector<ConstString> &CPP_names,
                                      const std::vector<ConstString> &C_names,
                                      const SymbolContext &sc);

  lldb::addr_t FindInSymbols(const std::vector<ConstString> &names,
                             const lldb_private::SymbolContext &sc,
                             bool &symbol_was_missing_weak);

  lldb::addr_t FindInRuntimes(const std::vector<ConstString> &names,
                              const lldb_private::SymbolContext &sc);

  lldb::addr_t FindInUserDefinedSymbols(const std::vector<ConstString> &names,
                                        const lldb_private::SymbolContext &sc);

  SymbolContext m_sym_ctx; ///< Used for symbol lookups

  bool m_strip_underscore; ///< True for platforms where global symbols have a
                           ///  _ prefix

  /// Any Module in this list will be used for symbol/function lookup before
  /// any other module (except for the module corresponding to the current
  /// frame).
  ModuleList m_preferred_modules;
};

} // namespace lldb_private

#endif // LLDB_EXPRESSION_EXTERNALSYMBOLRESOLVER_H
