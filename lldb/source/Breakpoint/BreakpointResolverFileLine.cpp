//===-- BreakpointResolverFileLine.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolverFileLine.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

// BreakpointResolverFileLine:
BreakpointResolverFileLine::BreakpointResolverFileLine(
    const BreakpointSP &bkpt, lldb::addr_t offset, bool skip_prologue,
    const SourceLocationSpec &location_spec)
    : BreakpointResolver(bkpt, BreakpointResolver::FileLineResolver, offset),
      m_location_spec(location_spec), m_skip_prologue(skip_prologue) {}

BreakpointResolver *BreakpointResolverFileLine::CreateFromStructuredData(
    const BreakpointSP &bkpt, const StructuredData::Dictionary &options_dict,
    Status &error) {
  llvm::StringRef filename;
  uint32_t line;
  uint16_t column;
  bool check_inlines;
  bool skip_prologue;
  bool exact_match;
  bool success;

  lldb::addr_t offset = 0;

  success = options_dict.GetValueForKeyAsString(GetKey(OptionNames::FileName),
                                                filename);
  if (!success) {
    error.SetErrorString("BRFL::CFSD: Couldn't find filename entry.");
    return nullptr;
  }

  success = options_dict.GetValueForKeyAsInteger(
      GetKey(OptionNames::LineNumber), line);
  if (!success) {
    error.SetErrorString("BRFL::CFSD: Couldn't find line number entry.");
    return nullptr;
  }

  success =
      options_dict.GetValueForKeyAsInteger(GetKey(OptionNames::Column), column);
  if (!success) {
    // Backwards compatibility.
    column = 0;
  }

  success = options_dict.GetValueForKeyAsBoolean(GetKey(OptionNames::Inlines),
                                                 check_inlines);
  if (!success) {
    error.SetErrorString("BRFL::CFSD: Couldn't find check inlines entry.");
    return nullptr;
  }

  success = options_dict.GetValueForKeyAsBoolean(
      GetKey(OptionNames::SkipPrologue), skip_prologue);
  if (!success) {
    error.SetErrorString("BRFL::CFSD: Couldn't find skip prologue entry.");
    return nullptr;
  }

  success = options_dict.GetValueForKeyAsBoolean(
      GetKey(OptionNames::ExactMatch), exact_match);
  if (!success) {
    error.SetErrorString("BRFL::CFSD: Couldn't find exact match entry.");
    return nullptr;
  }

  SourceLocationSpec location_spec(FileSpec(filename), line, column,
                                   check_inlines, exact_match);
  if (!location_spec)
    return nullptr;

  return new BreakpointResolverFileLine(bkpt, offset, skip_prologue,
                                        location_spec);
}

StructuredData::ObjectSP
BreakpointResolverFileLine::SerializeToStructuredData() {
  StructuredData::DictionarySP options_dict_sp(
      new StructuredData::Dictionary());

  options_dict_sp->AddBooleanItem(GetKey(OptionNames::SkipPrologue),
                                  m_skip_prologue);
  options_dict_sp->AddStringItem(GetKey(OptionNames::FileName),
                                 m_location_spec.GetFileSpec().GetPath());
  options_dict_sp->AddIntegerItem(GetKey(OptionNames::LineNumber),
                                  m_location_spec.GetLine().value_or(0));
  options_dict_sp->AddIntegerItem(
      GetKey(OptionNames::Column),
      m_location_spec.GetColumn().value_or(LLDB_INVALID_COLUMN_NUMBER));
  options_dict_sp->AddBooleanItem(GetKey(OptionNames::Inlines),
                                  m_location_spec.GetCheckInlines());
  options_dict_sp->AddBooleanItem(GetKey(OptionNames::ExactMatch),
                                  m_location_spec.GetExactMatch());

  return WrapOptionsDict(options_dict_sp);
}

// Filter the symbol context list to remove contexts where the line number was
// moved into a new function. We do this conservatively, so if e.g. we cannot
// resolve the function in the context (which can happen in case of line-table-
// only debug info), we leave the context as is. The trickiest part here is
// handling inlined functions -- in this case we need to make sure we look at
// the declaration line of the inlined function, NOT the function it was
// inlined into.
void BreakpointResolverFileLine::FilterContexts(SymbolContextList &sc_list,
                                                bool is_relative) {
  if (m_location_spec.GetExactMatch())
    return; // Nothing to do. Contexts are precise.

  llvm::StringRef relative_path;
  if (is_relative)
    relative_path = m_location_spec.GetFileSpec().GetDirectory().GetStringRef();

  Log *log = GetLog(LLDBLog::Breakpoints);
  for(uint32_t i = 0; i < sc_list.GetSize(); ++i) {
    SymbolContext sc;
    sc_list.GetContextAtIndex(i, sc);
    if (is_relative) {
      // If the path was relative, make sure any matches match as long as the
      // relative parts of the path match the path from support files
      auto sc_dir = sc.line_entry.file.GetDirectory().GetStringRef();
      if (!sc_dir.endswith(relative_path)) {
        // We had a relative path specified and the relative directory doesn't
        // match so remove this one
        LLDB_LOG(log, "removing not matching relative path {0} since it "
                "doesn't end with {1}", sc_dir, relative_path);
        sc_list.RemoveContextAtIndex(i);
        --i;
        continue;
      }
    }

    if (!sc.block)
      continue;

    FileSpec file;
    uint32_t line;
    const Block *inline_block = sc.block->GetContainingInlinedBlock();
    if (inline_block) {
      const Declaration &inline_declaration = inline_block->GetInlinedFunctionInfo()->GetDeclaration();
      if (!inline_declaration.IsValid())
        continue;
      file = inline_declaration.GetFile();
      line = inline_declaration.GetLine();
    } else if (sc.function)
      sc.function->GetStartLineSourceInfo(file, line);
    else
      continue;

    if (file != sc.line_entry.file) {
      LLDB_LOG(log, "unexpected symbol context file {0}", sc.line_entry.file);
      continue;
    }

    // Compare the requested line number with the line of the function
    // declaration. In case of a function declared as:
    //
    // int
    // foo()
    // {
    //   ...
    //
    // the compiler will set the declaration line to the "foo" line, which is
    // the reason why we have -1 here. This can fail in case of two inline
    // functions defined back-to-back:
    //
    // inline int foo1() { ... }
    // inline int foo2() { ... }
    //
    // but that's the best we can do for now.
    // One complication, if the line number returned from GetStartLineSourceInfo
    // is 0, then we can't do this calculation.  That can happen if
    // GetStartLineSourceInfo gets an error, or if the first line number in
    // the function really is 0 - which happens for some languages.

    // But only do this calculation if the line number we found in the SC
    // was different from the one requested in the source file.  If we actually
    // found an exact match it must be valid.

    if (m_location_spec.GetLine() == sc.line_entry.line)
      continue;

    const int decl_line_is_too_late_fudge = 1;
    if (line &&
        m_location_spec.GetLine() < line - decl_line_is_too_late_fudge) {
      LLDB_LOG(log, "removing symbol context at {0}:{1}", file, line);
      sc_list.RemoveContextAtIndex(i);
      --i;
    }
  }
}

Searcher::CallbackReturn BreakpointResolverFileLine::SearchCallback(
    SearchFilter &filter, SymbolContext &context, Address *addr) {
  SymbolContextList sc_list;

  // There is a tricky bit here.  You can have two compilation units that
  // #include the same file, and in one of them the function at m_line_number
  // is used (and so code and a line entry for it is generated) but in the
  // other it isn't.  If we considered the CU's independently, then in the
  // second inclusion, we'd move the breakpoint to the next function that
  // actually generated code in the header file.  That would end up being
  // confusing.  So instead, we do the CU iterations by hand here, then scan
  // through the complete list of matches, and figure out the closest line
  // number match, and only set breakpoints on that match.

  // Note also that if file_spec only had a file name and not a directory,
  // there may be many different file spec's in the resultant list.  The
  // closest line match for one will not be right for some totally different
  // file.  So we go through the match list and pull out the sets that have the
  // same file spec in their line_entry and treat each set separately.

  const uint32_t line = m_location_spec.GetLine().value_or(0);
  const llvm::Optional<uint16_t> column = m_location_spec.GetColumn();

  // We'll create a new SourceLocationSpec that can take into account the
  // relative path case, and we'll use it to resolve the symbol context
  // of the CUs.
  FileSpec search_file_spec = m_location_spec.GetFileSpec();
  const bool is_relative = search_file_spec.IsRelative();
  if (is_relative)
    search_file_spec.ClearDirectory();
  SourceLocationSpec search_location_spec(
      search_file_spec, m_location_spec.GetLine().value_or(0),
      m_location_spec.GetColumn(), m_location_spec.GetCheckInlines(),
      m_location_spec.GetExactMatch());

  const size_t num_comp_units = context.module_sp->GetNumCompileUnits();
  for (size_t i = 0; i < num_comp_units; i++) {
    CompUnitSP cu_sp(context.module_sp->GetCompileUnitAtIndex(i));
    if (cu_sp) {
      if (filter.CompUnitPasses(*cu_sp))
        cu_sp->ResolveSymbolContext(search_location_spec,
                                    eSymbolContextEverything, sc_list);
    }
  }

  FilterContexts(sc_list, is_relative);

  StreamString s;
  s.Printf("for %s:%d ",
           m_location_spec.GetFileSpec().GetFilename().AsCString("<Unknown>"),
           line);

  SetSCMatchesByLine(filter, sc_list, m_skip_prologue, s.GetString(), line,
                     column);

  return Searcher::eCallbackReturnContinue;
}

lldb::SearchDepth BreakpointResolverFileLine::GetDepth() {
  return lldb::eSearchDepthModule;
}

void BreakpointResolverFileLine::GetDescription(Stream *s) {
  s->Printf("file = '%s', line = %u, ",
            m_location_spec.GetFileSpec().GetPath().c_str(),
            m_location_spec.GetLine().value_or(0));
  auto column = m_location_spec.GetColumn();
  if (column)
    s->Printf("column = %u, ", *column);
  s->Printf("exact_match = %d", m_location_spec.GetExactMatch());
}

void BreakpointResolverFileLine::Dump(Stream *s) const {}

lldb::BreakpointResolverSP
BreakpointResolverFileLine::CopyForBreakpoint(BreakpointSP &breakpoint) {
  lldb::BreakpointResolverSP ret_sp(new BreakpointResolverFileLine(
      breakpoint, GetOffset(), m_skip_prologue, m_location_spec));

  return ret_sp;
}
