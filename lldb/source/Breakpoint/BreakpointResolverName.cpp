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

using namespace lldb;
using namespace lldb_private;

BreakpointResolverName::BreakpointResolverName
(
    Breakpoint *bkpt,
    const char *func_name,
    uint32_t func_name_type_mask,
    Breakpoint::MatchType type,
    bool skip_prologue
) :
    BreakpointResolver (bkpt, BreakpointResolver::NameResolver),
    m_func_name_type_mask (func_name_type_mask),
    m_class_name (),
    m_regex (),
    m_match_type (type),
    m_skip_prologue (skip_prologue)
{
    
    if (m_match_type == Breakpoint::Regexp)
    {
        if (!m_regex.Compile (func_name))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));

            if (log)
                log->Warning ("function name regexp: \"%s\" did not compile.", func_name);
        }
    }
    else
    {
        m_func_names.push_back(ConstString(func_name));
    }
}

BreakpointResolverName::BreakpointResolverName (Breakpoint *bkpt,
                                                const char *names[],
                                                size_t num_names,
                                                uint32_t name_type_mask,
                                                bool skip_prologue) :
    BreakpointResolver (bkpt, BreakpointResolver::NameResolver),
    m_func_name_type_mask (name_type_mask),
    m_match_type (Breakpoint::Exact),
    m_skip_prologue (skip_prologue)
{
    for (size_t i = 0; i < num_names; i++)
    {
        m_func_names.push_back (ConstString (names[i]));
    }
}

BreakpointResolverName::BreakpointResolverName (Breakpoint *bkpt,
                                                std::vector<std::string> names,
                                                uint32_t name_type_mask,
                                                bool skip_prologue) :
    BreakpointResolver (bkpt, BreakpointResolver::NameResolver),
    m_func_name_type_mask (name_type_mask),
    m_match_type (Breakpoint::Exact),
    m_skip_prologue (skip_prologue)
{
    size_t num_names = names.size();
    
    for (size_t i = 0; i < num_names; i++)
    {
        m_func_names.push_back (ConstString (names[i].c_str()));
    }
}

BreakpointResolverName::BreakpointResolverName
(
    Breakpoint *bkpt,
    RegularExpression &func_regex,
    bool skip_prologue
) :
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
    m_func_names.push_back(ConstString(method));
}

BreakpointResolverName::~BreakpointResolverName ()
{
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
    SymbolContext sc;
    Address break_addr;
    assert (m_breakpoint != NULL);
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    
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
                size_t num_names = m_func_names.size();
                for (int j = 0; j < num_names; j++)
                {
                    uint32_t num_functions = context.module_sp->FindFunctions (m_func_names[j], 
                                                                               NULL,
                                                                               m_func_name_type_mask, 
                                                                               include_symbols,
                                                                               include_inlines, 
                                                                               append, 
                                                                               func_list);
                    // If the search filter specifies a Compilation Unit, then we don't need to bother to look in plain
                    // symbols, since all the ones from a set compilation unit will have been found above already.
                    
                    if (num_functions == 0 && !filter_by_cu)
                    {
                        if (m_func_name_type_mask & (eFunctionNameTypeBase | eFunctionNameTypeFull | eFunctionNameTypeAuto))
                            context.module_sp->FindSymbolsWithNameAndType (m_func_names[j], eSymbolTypeCode, sym_list);
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
        size_t num_names = m_func_names.size();
        if (num_names == 1)
            s->Printf("name = '%s'", m_func_names[0].AsCString());
        else
        {
            s->Printf("names = {");
            for (size_t i = 0; i < num_names - 1; i++)
            {
                s->Printf ("'%s', ", m_func_names[i].AsCString());
            }
            s->Printf ("'%s'}", m_func_names[num_names - 1].AsCString());
        }
    }
}

void
BreakpointResolverName::Dump (Stream *s) const
{

}

