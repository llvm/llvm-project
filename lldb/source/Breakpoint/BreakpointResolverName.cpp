//===-- BreakpointResolverName.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolverName.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"

using namespace lldb;
using namespace lldb_private;

BreakpointResolverName::BreakpointResolverName (Breakpoint *bkpt,
                                                const char *name_cstr,
                                                uint32_t name_type_mask,
                                                Breakpoint::MatchType type,
                                                bool skip_prologue) :
    BreakpointResolver (bkpt, BreakpointResolver::NameResolver),
    m_class_name (),
    m_regex (),
    m_match_type (type),
    m_skip_prologue (skip_prologue)
{
    
    if (m_match_type == Breakpoint::Regexp)
    {
        if (!m_regex.Compile (name_cstr))
        {
            Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));

            if (log)
                log->Warning ("function name regexp: \"%s\" did not compile.", name_cstr);
        }
    }
    else
    {
        AddNameLookup (ConstString(name_cstr), name_type_mask);
    }
}

BreakpointResolverName::BreakpointResolverName (Breakpoint *bkpt,
                                                const char *names[],
                                                size_t num_names,
                                                uint32_t name_type_mask,
                                                bool skip_prologue) :
    BreakpointResolver (bkpt, BreakpointResolver::NameResolver),
    m_match_type (Breakpoint::Exact),
    m_skip_prologue (skip_prologue)
{
    for (size_t i = 0; i < num_names; i++)
    {
        AddNameLookup (ConstString (names[i]), name_type_mask);
    }
}

BreakpointResolverName::BreakpointResolverName (Breakpoint *bkpt,
                                                std::vector<std::string> names,
                                                uint32_t name_type_mask,
                                                bool skip_prologue) :
    BreakpointResolver (bkpt, BreakpointResolver::NameResolver),
    m_match_type (Breakpoint::Exact),
    m_skip_prologue (skip_prologue)
{
    for (const std::string& name : names)
    {
        AddNameLookup (ConstString (name.c_str(), name.size()), name_type_mask);
    }
}

BreakpointResolverName::BreakpointResolverName (Breakpoint *bkpt,
                                                RegularExpression &func_regex,
                                                bool skip_prologue) :
    BreakpointResolver (bkpt, BreakpointResolver::NameResolver),
    m_class_name (NULL),
    m_regex (func_regex),
    m_match_type (Breakpoint::Regexp),
    m_skip_prologue (skip_prologue)
{
}

BreakpointResolverName::BreakpointResolverName
(
    Breakpoint *bkpt,
    const char *class_name,
    const char *method,
    Breakpoint::MatchType type,
    bool skip_prologue
) :
    BreakpointResolver (bkpt, BreakpointResolver::NameResolver),
    m_class_name (class_name),
    m_regex (),
    m_match_type (type),
    m_skip_prologue (skip_prologue)
{
    LookupInfo lookup;
    lookup.name.SetCString(method);
    lookup.lookup_name = lookup.name;
    lookup.name_type_mask = eFunctionNameTypeMethod;
    lookup.match_name_after_lookup = false;
    m_lookups.push_back (lookup);
}

BreakpointResolverName::~BreakpointResolverName ()
{
}

void
BreakpointResolverName::AddNameLookup (const ConstString &name, uint32_t name_type_mask)
{
    ObjCLanguageRuntime::MethodName objc_method(name.GetCString(), false);
    if (objc_method.IsValid(false))
    {
        std::vector<ConstString> objc_names;
        objc_method.GetFullNames(objc_names, true);
        for (ConstString objc_name : objc_names)
        {
            LookupInfo lookup;
            lookup.name = name;
            lookup.lookup_name = objc_name;
            lookup.name_type_mask = eFunctionNameTypeFull;
            lookup.match_name_after_lookup = false;
            m_lookups.push_back (lookup);
        }
    }
    else
    {
        LookupInfo lookup;
        lookup.name = name;
        Module::PrepareForFunctionNameLookup(lookup.name, name_type_mask, lookup.lookup_name, lookup.name_type_mask, lookup.match_name_after_lookup);
        m_lookups.push_back (lookup);
    }
}


void
BreakpointResolverName::LookupInfo::Prune (SymbolContextList &sc_list, size_t start_idx) const
{
    if (match_name_after_lookup && name)
    {
        SymbolContext sc;
        size_t i = start_idx;
        while (i < sc_list.GetSize())
        {
            if (!sc_list.GetContextAtIndex(i, sc))
                break;
            ConstString full_name (sc.GetFunctionName());
            if (full_name && ::strstr(full_name.GetCString(), name.GetCString()) == NULL)
            {
                sc_list.RemoveContextAtIndex(i);
            }
            else
            {
                ++i;
            }
        }
    }
}


// FIXME: Right now we look at the module level, and call the module's "FindFunctions".
// Greg says he will add function tables, maybe at the CompileUnit level to accelerate function
// lookup.  At that point, we should switch the depth to CompileUnit, and look in these tables.

Searcher::CallbackReturn
BreakpointResolverName::SearchCallback
(
    SearchFilter &filter,
    SymbolContext &context,
    Address *addr,
    bool containing
)
{
    SymbolContextList func_list;
    SymbolContextList sym_list;
    
    uint32_t i;
    bool new_location;
    Address break_addr;
    assert (m_breakpoint != NULL);
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    
    if (m_class_name)
    {
        if (log)
            log->Warning ("Class/method function specification not supported yet.\n");
        return Searcher::eCallbackReturnStop;
    }
    
    const bool include_symbols = false;
    const bool include_inlines = true;
    const bool append = true;
    bool filter_by_cu = (filter.GetFilterRequiredItems() & eSymbolContextCompUnit) != 0;

    switch (m_match_type)
    {
        case Breakpoint::Exact:
            if (context.module_sp)
            {
                for (const LookupInfo &lookup : m_lookups)
                {
                    const size_t start_func_idx = func_list.GetSize();
                    context.module_sp->FindFunctions (lookup.lookup_name,
                                                      NULL,
                                                      lookup.name_type_mask,
                                                      include_symbols,
                                                      include_inlines,
                                                      append,
                                                      func_list);
                    const size_t end_func_idx = func_list.GetSize();

                    if (start_func_idx < end_func_idx)
                        lookup.Prune (func_list, start_func_idx);
                    // If the search filter specifies a Compilation Unit, then we don't need to bother to look in plain
                    // symbols, since all the ones from a set compilation unit will have been found above already.
                    else if (!filter_by_cu)
                    {
                        const size_t start_symbol_idx = sym_list.GetSize();
                        context.module_sp->FindFunctionSymbols (lookup.lookup_name, lookup.name_type_mask, sym_list);
                        const size_t end_symbol_idx = sym_list.GetSize();
                        if (start_symbol_idx < end_symbol_idx)
                            lookup.Prune (func_list, start_symbol_idx);
                    }
                }
            }
            break;
        case Breakpoint::Regexp:
            if (context.module_sp)
            {
                if (!filter_by_cu)
                    context.module_sp->FindSymbolsMatchingRegExAndType (m_regex, eSymbolTypeCode, sym_list);
                context.module_sp->FindFunctions (m_regex, 
                                                  include_symbols,
                                                  include_inlines, 
                                                  append, 
                                                  func_list);
            }
            break;
        case Breakpoint::Glob:
            if (log)
                log->Warning ("glob is not supported yet.");
            break;
    }

    // If the filter specifies a Compilation Unit, remove the ones that don't pass at this point.
    if (filter_by_cu)
    {
        uint32_t num_functions = func_list.GetSize();
        
        for (size_t idx = 0; idx < num_functions; idx++)
        {
            SymbolContext sc;
            func_list.GetContextAtIndex(idx, sc);
            if (!sc.comp_unit || !filter.CompUnitPasses(*sc.comp_unit))
            {
                func_list.RemoveContextAtIndex(idx);
                num_functions--;
                idx--;
            }
        }
    }

    // Remove any duplicates between the funcion list and the symbol list
    SymbolContext sc;
    if (func_list.GetSize())
    {
        for (i = 0; i < func_list.GetSize(); i++)
        {
            if (func_list.GetContextAtIndex(i, sc) == false)
                continue;
            
            if (sc.function == NULL)
                continue;
            uint32_t j = 0;
            while (j < sym_list.GetSize())
            {
                SymbolContext symbol_sc;
                if (sym_list.GetContextAtIndex(j, symbol_sc))
                {
                    if (symbol_sc.symbol && symbol_sc.symbol->ValueIsAddress())
                    {
                        if (sc.function->GetAddressRange().GetBaseAddress() == symbol_sc.symbol->GetAddress())
                        {
                            sym_list.RemoveContextAtIndex(j);
                            continue;   // Don't increment j
                        }
                    }
                }
                
                j++;
            }
        }
        
        for (i = 0; i < func_list.GetSize(); i++)
        {
            if (func_list.GetContextAtIndex(i, sc))
            {
                if (sc.block && sc.block->GetInlinedFunctionInfo())
                {
                    if (!sc.block->GetStartAddress(break_addr))
                        break_addr.Clear();
                }
                else if (sc.function)
                {
                    break_addr = sc.function->GetAddressRange().GetBaseAddress();
                    if (m_skip_prologue)
                    {
                        if (break_addr.IsValid())
                        {
                            const uint32_t prologue_byte_size = sc.function->GetPrologueByteSize();
                            if (prologue_byte_size)
                                break_addr.SetOffset(break_addr.GetOffset() + prologue_byte_size);
                        }
                    }
                }
                
                if (break_addr.IsValid())
                {
                    if (filter.AddressPasses(break_addr))
                    {
                        BreakpointLocationSP bp_loc_sp (m_breakpoint->AddLocation(break_addr, &new_location));
                        if (bp_loc_sp && new_location && !m_breakpoint->IsInternal())
                        {
                            if (log)
                            {
                                StreamString s;
                                bp_loc_sp->GetDescription(&s, lldb::eDescriptionLevelVerbose);
                                log->Printf ("Added location: %s\n", s.GetData());
                            }
                        }
                    }
                }
            }
        }
    }
    
    for (i = 0; i < sym_list.GetSize(); i++)
    {
        if (sym_list.GetContextAtIndex(i, sc))
        {
            if (sc.symbol && sc.symbol->ValueIsAddress())
            {
                break_addr = sc.symbol->GetAddress();
                
                if (m_skip_prologue)
                {
                    const uint32_t prologue_byte_size = sc.symbol->GetPrologueByteSize();
                    if (prologue_byte_size)
                        break_addr.SetOffset(break_addr.GetOffset() + prologue_byte_size);
                }
                
                if (filter.AddressPasses(break_addr))
                {
                    BreakpointLocationSP bp_loc_sp (m_breakpoint->AddLocation(break_addr, &new_location));
                    if (bp_loc_sp && new_location && !m_breakpoint->IsInternal())
                    {
                        StreamString s;
                        bp_loc_sp->GetDescription(&s, lldb::eDescriptionLevelVerbose);
                        if (log) 
                            log->Printf ("Added location: %s\n", s.GetData());
                    }
                }
            }
        }
    }
    return Searcher::eCallbackReturnContinue;
}

Searcher::Depth
BreakpointResolverName::GetDepth()
{
    return Searcher::eDepthModule;
}

void
BreakpointResolverName::GetDescription (Stream *s)
{
    if (m_match_type == Breakpoint::Regexp)
        s->Printf("regex = '%s'", m_regex.GetText());
    else
    {
        size_t num_names = m_lookups.size();
        if (num_names == 1)
            s->Printf("name = '%s'", m_lookups[0].name.GetCString());
        else
        {
            s->Printf("names = {");
            for (size_t i = 0; i < num_names - 1; i++)
            {
                s->Printf ("'%s', ", m_lookups[i].name.GetCString());
            }
            s->Printf ("'%s'}", m_lookups[num_names - 1].name.GetCString());
        }
    }
}

void
BreakpointResolverName::Dump (Stream *s) const
{

}

