//===-- TypeSynthetic.cpp ----------------------------------------*- C++ -*-===//
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
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

std::string
TypeFilterImpl::GetDescription()
{
    StreamString sstr;
    sstr.Printf("%s%s%s {\n",
                Cascades() ? "" : " (not cascading)",
                SkipsPointers() ? " (skip pointers)" : "",
                SkipsReferences() ? " (skip references)" : "");
    
    for (int i = 0; i < GetCount(); i++)
    {
        sstr.Printf("    %s\n",
                    GetExpressionPathAtIndex(i));
    }
    
    sstr.Printf("}");
    return sstr.GetString();
}

std::string
CXXSyntheticChildren::GetDescription()
{
    StreamString sstr;
    sstr.Printf("%s%s%s Generator at %p - %s",
                Cascades() ? "" : " (not cascading)",
                SkipsPointers() ? " (skip pointers)" : "",
                SkipsReferences() ? " (skip references)" : "",
                m_create_callback,
                m_description.c_str());
    
    return sstr.GetString();
}

#ifndef LLDB_DISABLE_PYTHON

ScriptedSyntheticChildren::FrontEnd::FrontEnd(std::string pclass, ValueObject &backend) :
SyntheticChildrenFrontEnd(backend),
m_python_class(pclass),
m_wrapper_sp(),
m_interpreter(NULL)
{
    if (backend == LLDB_INVALID_UID)
        return;
    
    TargetSP target_sp = backend.GetTargetSP();
    
    if (!target_sp)
        return;
    
    m_interpreter = target_sp->GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
    
    if (m_interpreter != NULL)
        m_wrapper_sp = m_interpreter->CreateSyntheticScriptedProvider(m_python_class.c_str(), backend.GetSP());
}

ScriptedSyntheticChildren::FrontEnd::~FrontEnd()
{
}

lldb::ValueObjectSP
ScriptedSyntheticChildren::FrontEnd::GetChildAtIndex (size_t idx)
{
    if (!m_wrapper_sp || !m_interpreter)
        return lldb::ValueObjectSP();
    
    return m_interpreter->GetChildAtIndex(m_wrapper_sp, idx);
}

std::string
ScriptedSyntheticChildren::GetDescription()
{
    StreamString sstr;
    sstr.Printf("%s%s%s Python class %s",
                Cascades() ? "" : " (not cascading)",
                SkipsPointers() ? " (skip pointers)" : "",
                SkipsReferences() ? " (skip references)" : "",
                m_python_class.c_str());
    
    return sstr.GetString();
}

#endif // #ifndef LLDB_DISABLE_PYTHON
