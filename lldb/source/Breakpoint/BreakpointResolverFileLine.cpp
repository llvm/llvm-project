//===-- BreakpointResolverFileLine.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolverFileLine.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// BreakpointResolverFileLine:
//----------------------------------------------------------------------
BreakpointResolverFileLine::BreakpointResolverFileLine(
    Breakpoint *bkpt, const FileSpec &file_spec, uint32_t line_no,
    lldb::addr_t offset, bool check_inlines, bool skip_prologue,
    bool exact_match)
    : BreakpointResolver(bkpt, BreakpointResolver::FileLineResolver, offset),
      m_file_spec(file_spec), m_line_number(line_no), m_inlines(check_inlines),
      m_skip_prologue(skip_prologue), m_exact_match(exact_match) {}

BreakpointResolverFileLine::~BreakpointResolverFileLine() {}

Searcher::CallbackReturn
BreakpointResolverFileLine::SearchCallback(SearchFilter &filter,
                                           SymbolContext &context,
                                           Address *addr, bool containing) {
  SymbolContextList sc_list;

  assert(m_breakpoint != NULL);

  // There is a tricky bit here.  You can have two compilation units that
  // #include the same file, and
  // in one of them the function at m_line_number is used (and so code and a
  // line entry for it is generated) but in the
  // other it isn't.  If we considered the CU's independently, then in the
  // second inclusion, we'd move the breakpoint
  // to the next function that actually generated code in the header file.  That
  // would end up being confusing.
  // So instead, we do the CU iterations by hand here, then scan through the
  // complete list of matches, and figure out
  // the closest line number match, and only set breakpoints on that match.

  // Note also that if file_spec only had a file name and not a directory, there
  // may be many different file spec's in
  // the resultant list.  The closest line match for one will not be right for
  // some totally different file.
  // So we go through the match list and pull out the sets that have the same
  // file spec in their line_entry
  // and treat each set separately.

  const size_t num_comp_units = context.module_sp->GetNumCompileUnits();
  for (size_t i = 0; i < num_comp_units; i++) {
    CompUnitSP cu_sp(context.module_sp->GetCompileUnitAtIndex(i));
    if (cu_sp) {
      if (filter.CompUnitPasses(*cu_sp))
        cu_sp->ResolveSymbolContext(m_file_spec, m_line_number, m_inlines,
                                    m_exact_match, eSymbolContextEverything,
                                    sc_list);
    }
  }
  StreamString s;
  s.Printf("for %s:%d ", m_file_spec.GetFilename().AsCString("<Unknown>"),
           m_line_number);

  SetSCMatchesByLine(filter, sc_list, m_skip_prologue, s.GetData());

  return Searcher::eCallbackReturnContinue;
}

Searcher::Depth BreakpointResolverFileLine::GetDepth() {
  return Searcher::eDepthModule;
}

void BreakpointResolverFileLine::GetDescription(Stream *s) {
  s->Printf("file = '%s', line = %u, exact_match = %d",
            m_file_spec.GetPath().c_str(), m_line_number, m_exact_match);
}

void BreakpointResolverFileLine::Dump(Stream *s) const {}

lldb::BreakpointResolverSP
BreakpointResolverFileLine::CopyForBreakpoint(Breakpoint &breakpoint) {
  lldb::BreakpointResolverSP ret_sp(new BreakpointResolverFileLine(
      &breakpoint, m_file_spec, m_line_number, m_offset, m_inlines,
      m_skip_prologue, m_exact_match));

  return ret_sp;
}
