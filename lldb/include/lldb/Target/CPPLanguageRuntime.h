//===-- CPPLanguageRuntime.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CPPLanguageRuntime_h_
#define liblldb_CPPLanguageRuntime_h_

// C Includes
// C++ Includes
#include <vector>
// Other libraries and framework includes
// Project includes
#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"
#include "lldb/Target/LanguageRuntime.h"

namespace lldb_private {

class CPPLanguageRuntime :
    public LanguageRuntime
{
public:
    
    class MethodName
    {
    public:
        enum Type
        {
            eTypeInvalid,
            eTypeUnknownMethod,
            eTypeClassMethod,
            eTypeInstanceMethod
        };
        
        MethodName () :
            m_full(),
            m_basename(),
            m_context(),
            m_arguments(),
            m_qualifiers(),
            m_type (eTypeInvalid),
            m_parsed (false),
            m_parse_error (false)
        {
        }

        MethodName (const ConstString &s) :
            m_full(s),
            m_basename(),
            m_context(),
            m_arguments(),
            m_qualifiers(),
            m_type (eTypeInvalid),
            m_parsed (false),
            m_parse_error (false)
        {
        }

        void
        Clear();
        
        bool
        IsValid () const
        {
            if (m_parse_error)
                return false;
            if (m_type == eTypeInvalid)
                return false;
            return (bool)m_full;
        }

        Type
        GetType () const
        {
            return m_type;
        }
        
        const ConstString &
        GetFullName () const
        {
            return m_full;
        }
        
        llvm::StringRef
        GetBasename ();

        llvm::StringRef
        GetContext ();
        
        llvm::StringRef
        GetArguments ();
        
        llvm::StringRef
        GetQualifiers ();

    protected:
        void
        Parse();

        ConstString     m_full;         // Full name:    "lldb::SBTarget::GetBreakpointAtIndex(unsigned int) const"
        llvm::StringRef m_basename;     // Basename:     "GetBreakpointAtIndex"
        llvm::StringRef m_context;      // Decl context: "lldb::SBTarget"
        llvm::StringRef m_arguments;    // Arguments:    "(unsigned int)"
        llvm::StringRef m_qualifiers;   // Qualifiers:   "const"
        Type m_type;
        bool m_parsed;
        bool m_parse_error;
    };

    virtual
    ~CPPLanguageRuntime();
    
    virtual lldb::LanguageType
    GetLanguageType () const
    {
        return lldb::eLanguageTypeC_plus_plus;
    }
    
    virtual bool
    IsVTableName (const char *name) = 0;
    
    virtual bool
    GetObjectDescription (Stream &str, ValueObject &object);
    
    virtual bool
    GetObjectDescription (Stream &str, Value &value, ExecutionContextScope *exe_scope);
    
    static bool
    IsCPPMangledName(const char *name);
    
    static bool
    StripNamespacesFromVariableName (const char *name, const char *&base_name_start, const char *&base_name_end);
    
    // in some cases, compilers will output different names for one same type. when tht happens, it might be impossible
    // to construct SBType objects for a valid type, because the name that is available is not the same as the name that
    // can be used as a search key in FindTypes(). the equivalents map here is meant to return possible alternative names
    // for a type through which a search can be conducted. Currently, this is only enabled for C++ but can be extended
    // to ObjC or other languages if necessary
    static uint32_t
    FindEquivalentNames(ConstString type_name, std::vector<ConstString>& equivalents);

protected:
    //------------------------------------------------------------------
    // Classes that inherit from CPPLanguageRuntime can see and modify these
    //------------------------------------------------------------------
    CPPLanguageRuntime(Process *process);
private:
    DISALLOW_COPY_AND_ASSIGN (CPPLanguageRuntime);
};

} // namespace lldb_private

#endif  // liblldb_CPPLanguageRuntime_h_
