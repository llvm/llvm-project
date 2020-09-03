//===-- BreakpointResolverFileLine.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_BREAKPOINT_BREAKPOINTRESOLVERFILELINE_H
#define LLDB_BREAKPOINT_BREAKPOINTRESOLVERFILELINE_H

#include "lldb/Breakpoint/BreakpointResolver.h"

namespace lldb_private {

/// \class BreakpointResolverFileLine BreakpointResolverFileLine.h
/// "lldb/Breakpoint/BreakpointResolverFileLine.h" This class sets breakpoints
/// by file and line.  Optionally, it will look for inlined instances of the
/// file and line specification.

class BreakpointResolverFileLine : public BreakpointResolver {
public:
  BreakpointResolverFileLine(const lldb::BreakpointSP &bkpt,
                             const FileSpec &resolver,
                             uint32_t line_no, uint32_t column,
                             lldb::addr_t m_offset, bool check_inlines,
                             bool skip_prologue, bool exact_match);

  static BreakpointResolver *
  CreateFromStructuredData(const lldb::BreakpointSP &bkpt,
                           const StructuredData::Dictionary &data_dict,
                           Status &error);

  StructuredData::ObjectSP SerializeToStructuredData() override;

  ~BreakpointResolverFileLine() override = default;

  Searcher::CallbackReturn SearchCallback(SearchFilter &filter,
                                          SymbolContext &context,
                                          Address *addr) override;

  lldb::SearchDepth GetDepth() override;

  void GetDescription(Stream *s) override;

  void Dump(Stream *s) const override;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BreakpointResolverFileLine *) {
    return true;
  }
  static inline bool classof(const BreakpointResolver *V) {
    return V->getResolverID() == BreakpointResolver::FileLineResolver;
  }

  lldb::BreakpointResolverSP
  CopyForBreakpoint(lldb::BreakpointSP &breakpoint) override;

protected:
  void FilterContexts(SymbolContextList &sc_list, bool is_relative);

  friend class Breakpoint;
  FileSpec m_file_spec;   ///< This is the file spec we are looking for.
  uint32_t m_line_number; ///< This is the line number that we are looking for.
  uint32_t m_column;      ///< This is the column that we are looking for.
  bool m_inlines; ///< This determines whether the resolver looks for inlined
                  ///< functions or not.
  bool m_skip_prologue;
  bool m_exact_match;

private:
  BreakpointResolverFileLine(const BreakpointResolverFileLine &) = delete;
  const BreakpointResolverFileLine &
  operator=(const BreakpointResolverFileLine &) = delete;
};

} // namespace lldb_private

#endif // LLDB_BREAKPOINT_BREAKPOINTRESOLVERFILELINE_H
