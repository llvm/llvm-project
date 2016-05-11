//===-- BreakpointResolver.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolver.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Address.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"

using namespace lldb_private;
using namespace lldb;

//----------------------------------------------------------------------
// BreakpointResolver:
//----------------------------------------------------------------------
BreakpointResolver::BreakpointResolver (Breakpoint *bkpt, const unsigned char resolverTy, lldb::addr_t offset) :
    m_breakpoint (bkpt),
    m_offset(offset),
    SubclassID (resolverTy)
{
}

BreakpointResolver::~BreakpointResolver ()
{

}

void
BreakpointResolver::SetBreakpoint (Breakpoint *bkpt)
{
    m_breakpoint = bkpt;
}

void
BreakpointResolver::ResolveBreakpointInModules (SearchFilter &filter, ModuleList &modules)
{
    filter.SearchInModuleList(*this, modules);
}

void
BreakpointResolver::ResolveBreakpoint (SearchFilter &filter)
{
    filter.Search (*this);
}

void
BreakpointResolver::SetSCMatchesByLine (SearchFilter &filter, SymbolContextList &sc_list, bool skip_prologue, const char *log_ident)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));

    while (sc_list.GetSize() > 0)
    {
        SymbolContextList tmp_sc_list;
        unsigned current_idx = 0;
        SymbolContext sc;
        bool first_entry = true;
        
        FileSpec match_file_spec;
        FileSpec match_original_file_spec;
        uint32_t closest_line_number = UINT32_MAX;

        // Pull out the first entry, and all the others that match its file spec, and stuff them in the tmp list.
        while (current_idx < sc_list.GetSize())
        {
            bool matches;
            
            sc_list.GetContextAtIndex (current_idx, sc);
            if (first_entry)
            {
                match_file_spec = sc.line_entry.file;
                match_original_file_spec = sc.line_entry.original_file;
                matches = true;
                first_entry = false;
            }
            else
                matches = ((sc.line_entry.file == match_file_spec) ||
                           (sc.line_entry.original_file == match_original_file_spec));
            
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
        
        current_idx = 0;
        std::map<Block *, lldb::addr_t> blocks_with_breakpoints;
        
        while (current_idx < tmp_sc_list.GetSize())
        {
            if (tmp_sc_list.GetContextAtIndex(current_idx, sc))
            {
                if (blocks_with_breakpoints.find (sc.block) != blocks_with_breakpoints.end())
                    tmp_sc_list.RemoveContextAtIndex(current_idx);
                else
                {
                    blocks_with_breakpoints.insert (std::pair<Block *, lldb::addr_t>(sc.block, sc.line_entry.range.GetBaseAddress().GetFileAddress()));
                    current_idx++;
                }
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
                        if (skip_prologue)
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
                    
                        BreakpointLocationSP bp_loc_sp (AddLocation(line_start));
                        if (log && bp_loc_sp && !m_breakpoint->IsInternal())
                        {
                            StreamString s;
                            bp_loc_sp->GetDescription (&s, lldb::eDescriptionLevelVerbose);
                            log->Printf ("Added location (skipped prologue: %s): %s \n", skipped_prologue ? "yes" : "no", s.GetData());
                        }
                    }
                    else if (log)
                    {
                        log->Printf ("Breakpoint %s at file address 0x%" PRIx64 " didn't pass the filter.\n",
                                     log_ident ? log_ident : "",
                                     line_start.GetFileAddress());
                    }
                }
                else
                {
                    if (log)
                        log->Printf ("error: Unable to set breakpoint %s at file address 0x%" PRIx64 "\n",
                                     log_ident ? log_ident : "",
                                     line_start.GetFileAddress());
                }
            }
        }
    }
}

BreakpointLocationSP
BreakpointResolver::AddLocation(Address loc_addr, bool *new_location)
{
    loc_addr.Slide(m_offset);
    return m_breakpoint->AddLocation(loc_addr, new_location);
}
    

void
BreakpointResolver::SetOffset (lldb::addr_t offset)
{
    // There may already be an offset, so we are actually adjusting location addresses by the difference.
    // lldb::addr_t slide = offset - m_offset;
    // FIXME: We should go fix up all the already set locations for the new slide.

    m_offset = offset;
}

