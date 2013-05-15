//===-- FormatClasses.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_FormatClasses_h_
#define lldb_FormatClasses_h_

// C Includes
#include <stdint.h>
#include <unistd.h>

// C++ Includes
#include <string>
#include <vector>

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Core/ValueObject.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/Type.h"

#include "lldb/DataFormatters/TypeFormat.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

namespace lldb_private {

class TypeNameSpecifierImpl
{
public:
    TypeNameSpecifierImpl() :
    m_is_regex(false),
    m_type()
    {
    }
    
    TypeNameSpecifierImpl (const char* name, bool is_regex) :
    m_is_regex(is_regex),
    m_type()
    {
        if (name)
            m_type.m_type_name.assign(name);
    }
    
    // if constructing with a given type, is_regex cannot be true since we are
    // giving an exact type to match
    TypeNameSpecifierImpl (lldb::TypeSP type) :
    m_is_regex(false),
    m_type()
    {
        if (type)
        {
            m_type.m_type_name.assign(type->GetName().GetCString());
            m_type.m_typeimpl_sp = lldb::TypeImplSP(new TypeImpl(type));
        }
    }

    TypeNameSpecifierImpl (ClangASTType type) :
    m_is_regex(false),
    m_type()
    {
        if (type.IsValid())
        {
            m_type.m_type_name.assign(type.GetConstTypeName().GetCString());
            m_type.m_typeimpl_sp = lldb::TypeImplSP(new TypeImpl(type));
        }
    }
    
    const char*
    GetName()
    {
        if (m_type.m_type_name.size())
            return m_type.m_type_name.c_str();
        return NULL;
    }
    
    lldb::TypeSP
    GetTypeSP ()
    {
        if (m_type.m_typeimpl_sp && m_type.m_typeimpl_sp->IsValid())
            return m_type.m_typeimpl_sp->GetTypeSP();
        return lldb::TypeSP();
    }
    
    ClangASTType
    GetClangASTType ()
    {
        if (m_type.m_typeimpl_sp && m_type.m_typeimpl_sp->IsValid())
            return m_type.m_typeimpl_sp->GetClangASTType();
        return ClangASTType();
    }
    
    bool
    IsRegex()
    {
        return m_is_regex;
    }
    
private:
    bool m_is_regex;
    // this works better than TypeAndOrName because the latter only wraps a TypeSP
    // whereas TypeImplSP can also be backed by a ClangASTType which is more commonly
    // used in LLDB. moreover, TypeImplSP is also what is currently backing SBType
    struct TypeOrName
    {
        std::string m_type_name;
        lldb::TypeImplSP m_typeimpl_sp;
    };
    TypeOrName m_type;
    
    
private:
    DISALLOW_COPY_AND_ASSIGN(TypeNameSpecifierImpl);
};
    
} // namespace lldb_private

#endif	// lldb_FormatClasses_h_
