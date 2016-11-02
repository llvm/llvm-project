//===-- BreakpointResolverFileRegex.cpp --------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolverFileRegex.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// BreakpointResolverFileRegex:
//----------------------------------------------------------------------
BreakpointResolverFileRegex::BreakpointResolverFileRegex(
    Breakpoint *bkpt, RegularExpression &regex,
    const std::unordered_set<std::string> &func_names, bool exact_match)
    : BreakpointResolver(bkpt, BreakpointResolver::FileLineResolver),
      m_regex(regex), m_exact_match(exact_match), m_function_names(func_names) {
}

BreakpointResolverFileRegex::~BreakpointResolverFileRegex() {}

Searcher::CallbackReturn
BreakpointResolverFileRegex::SearchCallback(SearchFilter &filter,
                                            SymbolContext &context,
                                            Address *addr, bool containing) {

  assert(m_breakpoint != NULL);
  if (!context.target_sp)
    return eCallbackReturnContinue;

  CompileUnit *cu = context.comp_unit;
  FileSpec cu_file_spec = *(static_cast<FileSpec *>(cu));
  std::vector<uint32_t> line_matches;
  context.target_sp->GetSourceManager().FindLinesMatchingRegex(
      cu_file_spec, m_regex, 1, UINT32_MAX, line_matches);

  uint32_t num_matches = line_matches.size();
  for (uint32_t i = 0; i < num_matches; i++) {
    SymbolContextList sc_list;
    const bool search_inlines = false;

    cu->ResolveSymbolContext(cu_file_spec, line_matches[i], search_inlines,
                             m_exact_match, eSymbolContextEverything, sc_list);
    // Find all the function names:
    if (!m_function_names.empty()) {
      std::vector<size_t> sc_to_remove;
      for (size_t i = 0; i < sc_list.GetSize(); i++) {
        SymbolContext sc_ctx;
        sc_list.GetContextAtIndex(i, sc_ctx);
        std::string name(
            sc_ctx
                .GetFunctionName(
                    Mangled::NamePreference::ePreferDemangledWithoutArguments)
                .AsCString());
        if (!m_function_names.count(name)) {
          sc_to_remove.push_back(i);
        }
      }

      if (!sc_to_remove.empty()) {
        std::vector<size_t>::reverse_iterator iter;
        std::vector<size_t>::reverse_iterator rend = sc_to_remove.rend();
        for (iter = sc_to_remove.rbegin(); iter != rend; iter++) {
          sc_list.RemoveContextAtIndex(*iter);
        }
      }
    }

    const bool skip_prologue = true;

    BreakpointResolver::SetSCMatchesByLine(filter, sc_list, skip_prologue,
                                           m_regex.GetText());
  }
  assert(m_breakpoint != NULL);

  return Searcher::eCallbackReturnContinue;
}

Searcher::Depth BreakpointResolverFileRegex::GetDepth() {
  return Searcher::eDepthCompUnit;
}

void BreakpointResolverFileRegex::GetDescription(Stream *s) {
  s->Printf("source regex = \"%s\", exact_match = %d", m_regex.GetText(),
            m_exact_match);
}

void BreakpointResolverFileRegex::Dump(Stream *s) const {}

lldb::BreakpointResolverSP
BreakpointResolverFileRegex::CopyForBreakpoint(Breakpoint &breakpoint) {
  lldb::BreakpointResolverSP ret_sp(new BreakpointResolverFileRegex(
      &breakpoint, m_regex, m_function_names, m_exact_match));
  return ret_sp;
}

void BreakpointResolverFileRegex::AddFunctionName(const char *func_name) {
  m_function_names.insert(func_name);
}
