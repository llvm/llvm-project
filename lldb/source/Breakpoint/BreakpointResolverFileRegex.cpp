//===-- BreakpointResolverFileRegex.cpp --------------------------*- C++ -*-===//
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
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Target/Target.h"
#include "lldb/lldb-private-log.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// BreakpointResolverFileRegex:
//----------------------------------------------------------------------
BreakpointResolverFileRegex::BreakpointResolverFileRegex
(
    Breakpoint *bkpt,
    RegularExpression &regex
) :
    BreakpointResolver (bkpt, BreakpointResolver::FileLineResolver),
    m_regex (regex)
{
}

BreakpointResolverFileRegex::~BreakpointResolverFileRegex ()
{
}

Searcher::CallbackReturn
BreakpointResolverFileRegex::SearchCallback
(
    SearchFilter &filter,
    SymbolContext &context,
    Address *addr,
    bool containing
)
{

    assert (m_breakpoint != NULL);
    if (!context.target_sp)
        return eCallbackReturnContinue;
        
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));

    CompileUnit *cu = context.comp_unit;
    FileSpec cu_file_spec = *(static_cast<FileSpec *>(cu));
    std::vector<uint32_t> line_matches;
    context.target_sp->GetSourceManager().FindLinesMatchingRegex(cu_file_spec, m_regex, 1, UINT32_MAX, line_matches); 
    uint32_t num_matches = line_matches.size();
    for (int i = 0; i < num_matches; i++)
    {
        uint32_t start_idx = 0;
        bool exact = false;
        while (1)
        {
            LineEntry line_entry;
        
            // Cycle through all the line entries that might match this one:
            start_idx = cu->FindLineEntry (start_idx, line_matches[i], NULL, exact, &line_entry);
            if (start_idx == UINT32_MAX)
                break;
            exact = true;
            start_idx++;
            
            Address line_start = line_entry.range.GetBaseAddress();
            if (line_start.IsValid())
            {
                if (filter.AddressPasses(line_start))
                {
                    BreakpointLocationSP bp_loc_sp (m_breakpoint->AddLocation(line_start));
                    if (log && bp_loc_sp && !m_breakpoint->IsInternal())
                    {
                        StreamString s;
                        bp_loc_sp->GetDescription (&s, lldb::eDescriptionLevelVerbose);
                        log->Printf ("Added location: %s\n", s.GetData());
                    }
                }
                else if (log)
                {
                    log->Printf ("Breakpoint at file address 0x%" PRIx64 " for %s:%d didn't pass filter.\n",
                                 line_start.GetFileAddress(),
                                 cu_file_spec.GetFilename().AsCString("<Unknown>"),
                                 line_matches[i]);
                }
            }
            else
            {
                if (log)
                    log->Printf ("error: Unable to set breakpoint at file address 0x%" PRIx64 " for %s:%d\n",
                                 line_start.GetFileAddress(),
                                 cu_file_spec.GetFilename().AsCString("<Unknown>"),
                                 line_matches[i]);
            }

        }
    }
    assert (m_breakpoint != NULL);        

    return Searcher::eCallbackReturnContinue;
}

Searcher::Depth
BreakpointResolverFileRegex::GetDepth()
{
    return Searcher::eDepthCompUnit;
}

void
BreakpointResolverFileRegex::GetDescription (Stream *s)
{
    s->Printf ("source regex = \"%s\"", m_regex.GetText());
}

void
BreakpointResolverFileRegex::Dump (Stream *s) const
{

}

