//===-- TypeSummary.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_TypeSummary_h_
#define lldb_TypeSummary_h_

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
#include "lldb/Symbol/Type.h"

namespace lldb_private {
    
    class TypeSummaryImpl
    {
    public:
        class Flags
        {
        public:
            
            Flags () :
            m_flags (lldb::eTypeOptionCascade)
            {}
            
            Flags (const Flags& other) :
            m_flags (other.m_flags)
            {}
            
            Flags (uint32_t value) :
            m_flags (value)
            {}
            
            Flags&
            operator = (const Flags& rhs)
            {
                if (&rhs != this)
                    m_flags = rhs.m_flags;
                
                return *this;
            }
            
            Flags&
            operator = (const uint32_t& rhs)
            {
                m_flags = rhs;
                return *this;
            }
            
            Flags&
            Clear()
            {
                m_flags = 0;
                return *this;
            }
            
            bool
            GetCascades () const
            {
                return (m_flags & lldb::eTypeOptionCascade) == lldb::eTypeOptionCascade;
            }
            
            Flags&
            SetCascades (bool value = true)
            {
                if (value)
                    m_flags |= lldb::eTypeOptionCascade;
                else
                    m_flags &= ~lldb::eTypeOptionCascade;
                return *this;
            }
            
            bool
            GetSkipPointers () const
            {
                return (m_flags & lldb::eTypeOptionSkipPointers) == lldb::eTypeOptionSkipPointers;
            }
            
            Flags&
            SetSkipPointers (bool value = true)
            {
                if (value)
                    m_flags |= lldb::eTypeOptionSkipPointers;
                else
                    m_flags &= ~lldb::eTypeOptionSkipPointers;
                return *this;
            }
            
            bool
            GetSkipReferences () const
            {
                return (m_flags & lldb::eTypeOptionSkipReferences) == lldb::eTypeOptionSkipReferences;
            }
            
            Flags&
            SetSkipReferences (bool value = true)
            {
                if (value)
                    m_flags |= lldb::eTypeOptionSkipReferences;
                else
                    m_flags &= ~lldb::eTypeOptionSkipReferences;
                return *this;
            }
            
            bool
            GetDontShowChildren () const
            {
                return (m_flags & lldb::eTypeOptionHideChildren) == lldb::eTypeOptionHideChildren;
            }
            
            Flags&
            SetDontShowChildren (bool value = true)
            {
                if (value)
                    m_flags |= lldb::eTypeOptionHideChildren;
                else
                    m_flags &= ~lldb::eTypeOptionHideChildren;
                return *this;
            }
            
            bool
            GetDontShowValue () const
            {
                return (m_flags & lldb::eTypeOptionHideValue) == lldb::eTypeOptionHideValue;
            }
            
            Flags&
            SetDontShowValue (bool value = true)
            {
                if (value)
                    m_flags |= lldb::eTypeOptionHideValue;
                else
                    m_flags &= ~lldb::eTypeOptionHideValue;
                return *this;
            }
            
            bool
            GetShowMembersOneLiner () const
            {
                return (m_flags & lldb::eTypeOptionShowOneLiner) == lldb::eTypeOptionShowOneLiner;
            }
            
            Flags&
            SetShowMembersOneLiner (bool value = true)
            {
                if (value)
                    m_flags |= lldb::eTypeOptionShowOneLiner;
                else
                    m_flags &= ~lldb::eTypeOptionShowOneLiner;
                return *this;
            }
            
            bool
            GetHideItemNames () const
            {
                return (m_flags & lldb::eTypeOptionHideNames) == lldb::eTypeOptionHideNames;
            }
            
            Flags&
            SetHideItemNames (bool value = true)
            {
                if (value)
                    m_flags |= lldb::eTypeOptionHideNames;
                else
                    m_flags &= ~lldb::eTypeOptionHideNames;
                return *this;
            }
            
            uint32_t
            GetValue ()
            {
                return m_flags;
            }
            
            void
            SetValue (uint32_t value)
            {
                m_flags = value;
            }
            
        private:
            uint32_t m_flags;
        };
        
        typedef enum Type
        {
            eTypeUnknown,
            eTypeString,
            eTypeScript,
            eTypeCallback
        } Type;
        
        TypeSummaryImpl (const TypeSummaryImpl::Flags& flags);
        
        bool
        Cascades () const
        {
            return m_flags.GetCascades();
        }
        bool
        SkipsPointers () const
        {
            return m_flags.GetSkipPointers();
        }
        bool
        SkipsReferences () const
        {
            return m_flags.GetSkipReferences();
        }
        
        bool
        DoesPrintChildren () const
        {
            return !m_flags.GetDontShowChildren();
        }
        
        bool
        DoesPrintValue () const
        {
            return !m_flags.GetDontShowValue();
        }
        
        bool
        IsOneliner () const
        {
            return m_flags.GetShowMembersOneLiner();
        }
        
        bool
        HideNames () const
        {
            return m_flags.GetHideItemNames();
        }
        
        void
        SetCascades (bool value)
        {
            m_flags.SetCascades(value);
        }
        
        void
        SetSkipsPointers (bool value)
        {
            m_flags.SetSkipPointers(value);
        }
        
        void
        SetSkipsReferences (bool value)
        {
            m_flags.SetSkipReferences(value);
        }
        
        void
        SetDoesPrintChildren (bool value)
        {
            m_flags.SetDontShowChildren(!value);
        }
        
        void
        SetDoesPrintValue (bool value)
        {
            m_flags.SetDontShowValue(!value);
        }
        
        void
        SetIsOneliner (bool value)
        {
            m_flags.SetShowMembersOneLiner(value);
        }
        
        void
        SetHideNames (bool value)
        {
            m_flags.SetHideItemNames(value);
        }
        
        uint32_t
        GetOptions ()
        {
            return m_flags.GetValue();
        }
        
        void
        SetOptions (uint32_t value)
        {
            m_flags.SetValue(value);
        }
        
        virtual
        ~TypeSummaryImpl ()
        {
        }
        
        // we are using a ValueObject* instead of a ValueObjectSP because we do not need to hold on to this for
        // extended periods of time and we trust the ValueObject to stay around for as long as it is required
        // for us to generate its summary
        virtual bool
        FormatObject (ValueObject *valobj,
                      std::string& dest) = 0;
        
        virtual std::string
        GetDescription () = 0;
        
        virtual bool
        IsScripted () = 0;
        
        virtual Type
        GetType () = 0;
        
        uint32_t&
        GetRevision ()
        {
            return m_my_revision;
        }
        
        typedef std::shared_ptr<TypeSummaryImpl> SharedPointer;
        typedef bool(*SummaryCallback)(void*, ConstString, const lldb::TypeSummaryImplSP&);
        typedef bool(*RegexSummaryCallback)(void*, lldb::RegularExpressionSP, const lldb::TypeSummaryImplSP&);
        
    protected:
        uint32_t m_my_revision;
        Flags m_flags;
        
    private:
        DISALLOW_COPY_AND_ASSIGN(TypeSummaryImpl);
    };
    
    // simple string-based summaries, using ${var to show data
    struct StringSummaryFormat : public TypeSummaryImpl
    {
        std::string m_format;
        
        StringSummaryFormat(const TypeSummaryImpl::Flags& flags,
                            const char* f);
        
        const char*
        GetSummaryString () const
        {
            return m_format.c_str();
        }
        
        void
        SetSummaryString (const char* data)
        {
            if (data)
                m_format.assign(data);
            else
                m_format.clear();
        }
        
        virtual
        ~StringSummaryFormat()
        {
        }
        
        virtual bool
        FormatObject(ValueObject *valobj,
                     std::string& dest);
        
        virtual std::string
        GetDescription();
        
        virtual bool
        IsScripted ()
        {
            return false;
        }
        
        
        virtual Type
        GetType ()
        {
            return TypeSummaryImpl::eTypeString;
        }
        
    private:
        DISALLOW_COPY_AND_ASSIGN(StringSummaryFormat);
    };
    
    // summaries implemented via a C++ function
    struct CXXFunctionSummaryFormat : public TypeSummaryImpl
    {
        
        // we should convert these to SBValue and SBStream if we ever cross
        // the boundary towards the external world
        typedef bool (*Callback)(ValueObject& valobj, Stream& dest);
        
        Callback m_impl;
        std::string m_description;
        
        CXXFunctionSummaryFormat (const TypeSummaryImpl::Flags& flags,
                                  Callback impl,
                                  const char* description);
        
        Callback
        GetBackendFunction () const
        {
            return m_impl;
        }
        
        const char*
        GetTextualInfo () const
        {
            return m_description.c_str();
        }
        
        void
        SetBackendFunction (Callback cb_func)
        {
            m_impl = cb_func;
        }
        
        void
        SetTextualInfo (const char* descr)
        {
            if (descr)
                m_description.assign(descr);
            else
                m_description.clear();
        }
        
        virtual
        ~CXXFunctionSummaryFormat ()
        {
        }
        
        virtual bool
        FormatObject (ValueObject *valobj,
                      std::string& dest);
        
        virtual std::string
        GetDescription ();
        
        virtual bool
        IsScripted ()
        {
            return false;
        }
        
        virtual Type
        GetType ()
        {
            return TypeSummaryImpl::eTypeCallback;
        }
        
        typedef std::shared_ptr<CXXFunctionSummaryFormat> SharedPointer;
        
    private:
        DISALLOW_COPY_AND_ASSIGN(CXXFunctionSummaryFormat);
    };
    
#ifndef LLDB_DISABLE_PYTHON
    
    // Python-based summaries, running script code to show data
    struct ScriptSummaryFormat : public TypeSummaryImpl
    {
        std::string m_function_name;
        std::string m_python_script;
        lldb::ScriptInterpreterObjectSP m_script_function_sp;
        
        ScriptSummaryFormat(const TypeSummaryImpl::Flags& flags,
                            const char *function_name,
                            const char* python_script = NULL);
        
        const char*
        GetFunctionName () const
        {
            return m_function_name.c_str();
        }
        
        const char*
        GetPythonScript () const
        {
            return m_python_script.c_str();
        }
        
        void
        SetFunctionName (const char* function_name)
        {
            if (function_name)
                m_function_name.assign(function_name);
            else
                m_function_name.clear();
            m_python_script.clear();
        }
        
        void
        SetPythonScript (const char* script)
        {
            if (script)
                m_python_script.assign(script);
            else
                m_python_script.clear();
        }
        
        virtual
        ~ScriptSummaryFormat ()
        {
        }
        
        virtual bool
        FormatObject (ValueObject *valobj,
                      std::string& dest);
        
        virtual std::string
        GetDescription ();
        
        virtual bool
        IsScripted ()
        {
            return true;
        }
        
        virtual Type
        GetType ()
        {
            return TypeSummaryImpl::eTypeScript;
        }
        
        typedef std::shared_ptr<ScriptSummaryFormat> SharedPointer;
        
        
    private:
        DISALLOW_COPY_AND_ASSIGN(ScriptSummaryFormat);
    };
#endif
} // namespace lldb_private

#endif	// lldb_TypeSummary_h_
