//===-- TypeSummary.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

// C Includes

// C++ Includes

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

#include "lldb/Host/Host.h"

using namespace lldb;
using namespace lldb_private;

TypeSummaryImpl::TypeSummaryImpl (const TypeSummaryImpl::Flags& flags) :
m_flags(flags)
{
}


StringSummaryFormat::StringSummaryFormat (const TypeSummaryImpl::Flags& flags,
                                          const char *format_cstr) :
TypeSummaryImpl(flags),
m_format()
{
    if (format_cstr)
        m_format.assign(format_cstr);
}

bool
StringSummaryFormat::FormatObject (ValueObject *valobj,
                                   std::string& retval)
{
    if (!valobj)
    {
        retval.assign("NULL ValueObject");
        return false;
    }
    
    StreamString s;
    ExecutionContext exe_ctx (valobj->GetExecutionContextRef());
    SymbolContext sc;
    StackFrame *frame = exe_ctx.GetFramePtr();
    if (frame)
        sc = frame->GetSymbolContext(lldb::eSymbolContextEverything);
    
    if (IsOneliner())
    {
        ValueObject* object;
        
        ValueObjectSP synth_valobj = valobj->GetSyntheticValue();
        if (synth_valobj)
            object = synth_valobj.get();
        else
            object = valobj;
        
        const uint32_t num_children = object->GetNumChildren();
        if (num_children)
        {
            s.PutChar('(');
            
            for (uint32_t idx=0; idx<num_children; ++idx)
            {
                lldb::ValueObjectSP child_sp(object->GetChildAtIndex(idx, true));
                if (child_sp.get())
                {
                    if (idx)
                        s.PutCString(", ");
                    if (!HideNames())
                    {
                        s.PutCString(child_sp.get()->GetName().AsCString());
                        s.PutCString(" = ");
                    }
                    child_sp.get()->DumpPrintableRepresentation(s,
                                                                ValueObject::eValueObjectRepresentationStyleSummary,
                                                                lldb::eFormatInvalid,
                                                                ValueObject::ePrintableRepresentationSpecialCasesDisable);
                }
            }
            
            s.PutChar(')');
            
            retval.assign(s.GetString());
            return true;
        }
        else
        {
            retval.assign("error: oneliner for no children");
            return false;
        }
        
    }
    else
    {
        if (Debugger::FormatPrompt(m_format.c_str(), &sc, &exe_ctx, &sc.line_entry.range.GetBaseAddress(), s, valobj))
        {
            retval.assign(s.GetString());
            return true;
        }
        else
        {
            retval.assign("error: summary string parsing error");
            return false;
        }
    }
}

std::string
StringSummaryFormat::GetDescription ()
{
    StreamString sstr;
    
    sstr.Printf ("`%s`%s%s%s%s%s%s%s",      m_format.c_str(),
                 Cascades() ? "" : " (not cascading)",
                 !DoesPrintChildren() ? "" : " (show children)",
                 !DoesPrintValue() ? " (hide value)" : "",
                 IsOneliner() ? " (one-line printout)" : "",
                 SkipsPointers() ? " (skip pointers)" : "",
                 SkipsReferences() ? " (skip references)" : "",
                 HideNames() ? " (hide member names)" : "");
    return sstr.GetString();
}

CXXFunctionSummaryFormat::CXXFunctionSummaryFormat (const TypeSummaryImpl::Flags& flags,
                                                    Callback impl,
                                                    const char* description) :
TypeSummaryImpl(flags),
m_impl(impl),
m_description(description ? description : "")
{
}

bool
CXXFunctionSummaryFormat::FormatObject (ValueObject *valobj,
                                        std::string& dest)
{
    dest.clear();
    StreamString stream;
    if (!m_impl || m_impl(*valobj,stream) == false)
        return false;
    dest.assign(stream.GetData());
    return true;
}

std::string
CXXFunctionSummaryFormat::GetDescription ()
{
    StreamString sstr;
    sstr.Printf ("`%s (%p) `%s%s%s%s%s%s%s",      m_description.c_str(),m_impl,
                 Cascades() ? "" : " (not cascading)",
                 !DoesPrintChildren() ? "" : " (show children)",
                 !DoesPrintValue() ? " (hide value)" : "",
                 IsOneliner() ? " (one-line printout)" : "",
                 SkipsPointers() ? " (skip pointers)" : "",
                 SkipsReferences() ? " (skip references)" : "",
                 HideNames() ? " (hide member names)" : "");
    return sstr.GetString();
}

#ifndef LLDB_DISABLE_PYTHON


ScriptSummaryFormat::ScriptSummaryFormat (const TypeSummaryImpl::Flags& flags,
                                          const char * function_name,
                                          const char * python_script) :
TypeSummaryImpl(flags),
m_function_name(),
m_python_script(),
m_script_function_sp()
{
    if (function_name)
        m_function_name.assign(function_name);
    if (python_script)
        m_python_script.assign(python_script);
}

bool
ScriptSummaryFormat::FormatObject (ValueObject *valobj,
                                   std::string& retval)
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
    
    if (!valobj)
        return false;
    
    Host::SetCrashDescriptionWithFormat("[Python summary] Name: %s - Function: %s",
                                        valobj->GetName().AsCString("unknown"),
                                        m_function_name.c_str());

    TargetSP target_sp(valobj->GetTargetSP());
    
    if (!target_sp)
    {
        retval.assign("error: no target");
        return false;
    }
    
    ScriptInterpreter *script_interpreter = target_sp->GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
    
    if (!script_interpreter)
    {
        retval.assign("error: no ScriptInterpreter");
        return false;
    }
    
    return script_interpreter->GetScriptedSummary(m_function_name.c_str(),
                                                  valobj->GetSP(),
                                                  m_script_function_sp,
                                                  retval);
    
}

std::string
ScriptSummaryFormat::GetDescription ()
{
    StreamString sstr;
    sstr.Printf ("%s%s%s%s%s%s%s\n%s",       Cascades() ? "" : " (not cascading)",
                 !DoesPrintChildren() ? "" : " (show children)",
                 !DoesPrintValue() ? " (hide value)" : "",
                 IsOneliner() ? " (one-line printout)" : "",
                 SkipsPointers() ? " (skip pointers)" : "",
                 SkipsReferences() ? " (skip references)" : "",
                 HideNames() ? " (hide member names)" : "",
                 m_python_script.c_str());
    return sstr.GetString();
    
}

#endif // #ifndef LLDB_DISABLE_PYTHON
