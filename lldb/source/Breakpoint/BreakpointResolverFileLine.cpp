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
#include "lldb/lldb-private-log.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// BreakpointResolverFileLine:
//----------------------------------------------------------------------
BreakpointResolverFileLine::BreakpointResolverFileLine
(
    Breakpoint *bkpt,
    const FileSpec &file_spec,
    uint32_t line_no,
    bool check_inlines,
    bool skip_prologue
) :
    BreakpointResolver (bkpt, BreakpointResolver::FileLineResolver),
    m_file_spec (file_spec),
    m_line_number (line_no),
    m_inlines (check_inlines),
    m_skip_prologue(skip_prologue)
{
}

BreakpointResolverFileLine::~BreakpointResolverFileLine ()
{
}

Searcher::CallbackReturn
BreakpointResolverFileLine::SearchCallback
(
    SearchFilter &filter,
    SymbolContext &context,
    Address *addr,
    bool containing
)
{
    SymbolContextList sc_list;

    assert (m_breakpoint != NULL);
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    
    // There is a tricky bit here.  You can have two compilation units that #include the same file, and
    // in one of them the function at m_line_number is used (and so code and a line entry for it is generated) but in the
    // other it isn't.  If we considered the CU's independently, then in the second inclusion, we'd move the breakpoint 
    // to the next function that actually generated code in the header file.  That would end up being confusing.
    // So instead, we do the CU iterations by hand here, then scan through the complete list of matches, and figure out 
    // the closest line number match, and only set breakpoints on that match.
    
    // Note also that if file_spec only had a file name and not a directory, there may be many different file spec's in
    // the resultant list.  The closest line match for one will not be right for some totally different file.
    // So we go through the match list and pull out the sets that have the same file spec in their line_entry
    // and treat each set separately.
    
    const size_t num_comp_units = context.module_sp->GetNumCompileUnits();
    for (size_t i = 0; i < num_comp_units; i++)
    {
        CompUnitSP cu_sp (context.module_sp->GetCompileUnitAtIndex (i));
        if (cu_sp)
        {
            if (filter.CompUnitPasses(*cu_sp))
                cu_sp->ResolveSymbolContext (m_file_spec, m_line_number, m_inlines, false, eSymbolContextEverything, sc_list);
        }
    }
    
    while (sc_list.GetSize() > 0)
    {
        SymbolContextList tmp_sc_list;
        int current_idx = 0;
        SymbolContext sc;
        bool first_entry = true;
        
        FileSpec match_file_spec;
        uint32_t closest_line_number = UINT32_MAX;

        // Pull out the first entry, and all the others that match its file spec, and stuff them in the tmp list.
        while (current_idx < sc_list.GetSize())
        {
            bool matches;
            
            sc_list.GetContextAtIndex (current_idx, sc);
            if (first_entry)
            {
                match_file_spec = sc.line_entry.file;
                matches = true;
                first_entry = false;
            }
            else
                matches = (sc.line_entry.file == match_file_spec);
            
            if (matches)
            {
                tmp_sc_list.Append (sc);
                sc_list.RemoveContextAtIndex(current_idx);
                
                // ResolveSymbolContext will always return a number that is >= the line number you pass in.
                // So the smaller line number is always better.
                if (sc.line_entry.line < closest_line_number)
                    closest_line_number = sc.line_entry.line;
            }
            else
                current_idx++;
        }
            
        // Okay, we've found the closest line number match, now throw away all the others:
        
        current_idx = 0;
        while (current_idx < tmp_sc_list.GetSize())
        {
            if (tmp_sc_list.GetContextAtIndex(current_idx, sc))
            {
                if (sc.line_entry.line != closest_line_number)
                    tmp_sc_list.RemoveContextAtIndex(current_idx);
                else
                    current_idx++;
            }
        }
        
        // Next go through and see if there are line table entries that are contiguous, and if so keep only the
        // first of the contiguous range:
        
        lldb::addr_t last_end_addr = LLDB_INVALID_ADDRESS;
        current_idx = 0;
        while (current_idx < tmp_sc_list.GetSize())
        {
            if (tmp_sc_list.GetContextAtIndex(current_idx, sc))
            {
                lldb::addr_t start_file_addr = sc.line_entry.range.GetBaseAddress().GetFileAddress();
                lldb::addr_t end_file_addr   = start_file_addr + sc.line_entry.range.GetByteSize();
                
                if (start_file_addr == last_end_addr)
                    tmp_sc_list.RemoveContextAtIndex(current_idx);
                else
                    current_idx++;

                last_end_addr = end_file_addr;
            }
        }
        
        // and make breakpoints out of the closest line number match.
        
        uint32_t tmp_sc_list_size = tmp_sc_list.GetSize();
        
        for (uint32_t i = 0; i < tmp_sc_list_size; i++)
        {
            if (tmp_sc_list.GetContextAtIndex(i, sc))
            {
                Address line_start = sc.line_entry.range.GetBaseAddress();
                if (line_start.IsValid())
                {
                    if (filter.AddressPasses(line_start))
                    {
                        // If the line number is before the prologue end, move it there...
                        bool skipped_prologue = false;
                        if (m_skip_prologue)
                        {
                            if (sc.function)
                            {
                                Address prologue_addr(sc.function->GetAddressRange().GetBaseAddress());
                                if (prologue_addr.IsValid() && (line_start == prologue_addr))
                                {
                                    const uint32_t prologue_byte_size = sc.function->GetPrologueByteSize();
                                    if (prologue_byte_size)
                                    {
                                        prologue_addr.Slide(prologue_byte_size);
                 
                                        if (filter.AddressPasses(prologue_addr))
                                        {
                                            skipped_prologue = true;
                                            line_start = prologue_addr;
                                        }
                                    }
                                }
                            }
                        }
                    
                        BreakpointLocationSP bp_loc_sp (m_breakpoint->AddLocation(line_start));
                        if (log && bp_loc_sp && !m_breakpoint->IsInternal())
                        {
                            StreamString s;
                            bp_loc_sp->GetDescription (&s, lldb::eDescriptionLevelVerbose);
                            log->Printf ("Added location (skipped prologue: %s): %s \n", skipped_prologue ? "yes" : "no", s.GetData());
                        }
                    }
                    else if (log)
                    {
                        log->Printf ("Breakpoint at file address 0x%" PRIx64 " for %s:%d didn't pass the filter.\n",
                                     line_start.GetFileAddress(),
                                     m_file_spec.GetFilename().AsCString("<Unknown>"),
                                     m_line_number);
                    }
                }
                else
                {
                    if (log)
                        log->Printf ("error: Unable to set breakpoint at file address 0x%" PRIx64 " for %s:%d\n",
                                     line_start.GetFileAddress(),
                                     m_file_spec.GetFilename().AsCString("<Unknown>"),
                                     m_line_number);
                }
            }
        }
    }
        
    return Searcher::eCallbackReturnContinue;
}

Searcher::Depth
BreakpointResolverFileLine::GetDepth()
{
    return Searcher::eDepthModule;
}

void
BreakpointResolverFileLine::GetDescription (Stream *s)
{
    s->Printf ("file ='%s', line = %u", m_file_spec.GetFilename().AsCString(), m_line_number);
}

void
BreakpointResolverFileLine::Dump (Stream *s) const
{

}

