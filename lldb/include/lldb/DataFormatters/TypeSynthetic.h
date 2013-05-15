//===-- TypeSynthetic.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_TypeSynthetic_h_
#define lldb_TypeSynthetic_h_

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
    class SyntheticChildrenFrontEnd
    {
    protected:
        ValueObject &m_backend;
    public:
        
        SyntheticChildrenFrontEnd (ValueObject &backend) :
        m_backend(backend)
        {}
        
        virtual
        ~SyntheticChildrenFrontEnd ()
        {
        }
        
        virtual size_t
        CalculateNumChildren () = 0;
        
        virtual lldb::ValueObjectSP
        GetChildAtIndex (size_t idx) = 0;
        
        virtual size_t
        GetIndexOfChildWithName (const ConstString &name) = 0;
        
        // this function is assumed to always succeed and it if fails, the front-end should know to deal
        // with it in the correct way (most probably, by refusing to return any children)
        // the return value of Update() should actually be interpreted as "ValueObjectSyntheticFilter cache is good/bad"
        // if =true, ValueObjectSyntheticFilter is allowed to use the children it fetched previously and cached
        // if =false, ValueObjectSyntheticFilter must throw away its cache, and query again for children
        virtual bool
        Update () = 0;
        
        // if this function returns false, then CalculateNumChildren() MUST return 0 since UI frontends
        // might validly decide not to inquire for children given a false return value from this call
        // if it returns true, then CalculateNumChildren() can return any number >= 0 (0 being valid)
        // it should if at all possible be more efficient than CalculateNumChildren()
        virtual bool
        MightHaveChildren () = 0;
        
        typedef std::shared_ptr<SyntheticChildrenFrontEnd> SharedPointer;
        typedef std::unique_ptr<SyntheticChildrenFrontEnd> AutoPointer;
        
    private:
        DISALLOW_COPY_AND_ASSIGN(SyntheticChildrenFrontEnd);
    };
    
    class SyntheticChildren
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
        
        SyntheticChildren (const Flags& flags) :
        m_flags(flags)
        {
        }
        
        virtual
        ~SyntheticChildren ()
        {
        }
        
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
        
        virtual bool
        IsScripted () = 0;
        
        virtual std::string
        GetDescription () = 0;
        
        virtual SyntheticChildrenFrontEnd::AutoPointer
        GetFrontEnd (ValueObject &backend) = 0;
        
        typedef std::shared_ptr<SyntheticChildren> SharedPointer;
        typedef bool(*SyntheticChildrenCallback)(void*, ConstString, const SyntheticChildren::SharedPointer&);
        
        uint32_t&
        GetRevision ()
        {
            return m_my_revision;
        }
        
    protected:
        uint32_t m_my_revision;
        Flags m_flags;
        
    private:
        DISALLOW_COPY_AND_ASSIGN(SyntheticChildren);
    };
    
    class TypeFilterImpl : public SyntheticChildren
    {
        std::vector<std::string> m_expression_paths;
    public:
        TypeFilterImpl(const SyntheticChildren::Flags& flags) :
        SyntheticChildren(flags),
        m_expression_paths()
        {
        }

        TypeFilterImpl(const SyntheticChildren::Flags& flags,
                       const std::initializer_list<const char*> items) :
        SyntheticChildren(flags),
        m_expression_paths()
        {
            for (auto path : items)
                AddExpressionPath (path);
        }
        
        void
        AddExpressionPath (const char* path)
        {
            AddExpressionPath(std::string(path));
        }
        
        void
        Clear()
        {
            m_expression_paths.clear();
        }
        
        int
        GetCount() const
        {
            return m_expression_paths.size();
        }
        
        const char*
        GetExpressionPathAtIndex(int i) const
        {
            return m_expression_paths[i].c_str();
        }
        
        bool
        SetExpressionPathAtIndex (int i, const char* path)
        {
            return SetExpressionPathAtIndex(i, std::string(path));
        }
        
        void
        AddExpressionPath (const std::string& path)
        {
            bool need_add_dot = true;
            if (path[0] == '.' ||
                (path[0] == '-' && path[1] == '>') ||
                path[0] == '[')
                need_add_dot = false;
            // add a '.' symbol to help forgetful users
            if(!need_add_dot)
                m_expression_paths.push_back(path);
            else
                m_expression_paths.push_back(std::string(".") + path);
        }
        
        bool
        SetExpressionPathAtIndex (int i, const std::string& path)
        {
            if (i >= GetCount())
                return false;
            bool need_add_dot = true;
            if (path[0] == '.' ||
                (path[0] == '-' && path[1] == '>') ||
                path[0] == '[')
                need_add_dot = false;
            // add a '.' symbol to help forgetful users
            if(!need_add_dot)
                m_expression_paths[i] = path;
            else
                m_expression_paths[i] = std::string(".") + path;
            return true;
        }
        
        bool
        IsScripted ()
        {
            return false;
        }
        
        std::string
        GetDescription ();
        
        class FrontEnd : public SyntheticChildrenFrontEnd
        {
        private:
            TypeFilterImpl* filter;
        public:
            
            FrontEnd(TypeFilterImpl* flt,
                     ValueObject &backend) :
            SyntheticChildrenFrontEnd(backend),
            filter(flt)
            {}
            
            virtual
            ~FrontEnd ()
            {
            }
            
            virtual size_t
            CalculateNumChildren ()
            {
                return filter->GetCount();
            }
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx)
            {
                if (idx >= filter->GetCount())
                    return lldb::ValueObjectSP();
                return m_backend.GetSyntheticExpressionPathChild(filter->GetExpressionPathAtIndex(idx), true);
            }
            
            virtual bool
            Update() { return false; }
            
            virtual bool
            MightHaveChildren ()
            {
                return filter->GetCount() > 0;
            }
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name)
            {
                const char* name_cstr = name.GetCString();
                for (int i = 0; i < filter->GetCount(); i++)
                {
                    const char* expr_cstr = filter->GetExpressionPathAtIndex(i);
                    if (expr_cstr)
                    {
                        if (*expr_cstr == '.')
                            expr_cstr++;
                        else if (*expr_cstr == '-' && *(expr_cstr+1) == '>')
                            expr_cstr += 2;
                    }
                    if (!::strcmp(name_cstr, expr_cstr))
                        return i;
                }
                return UINT32_MAX;
            }
            
            typedef std::shared_ptr<SyntheticChildrenFrontEnd> SharedPointer;
            
        private:
            DISALLOW_COPY_AND_ASSIGN(FrontEnd);
        };
        
        virtual SyntheticChildrenFrontEnd::AutoPointer
        GetFrontEnd(ValueObject &backend)
        {
            return SyntheticChildrenFrontEnd::AutoPointer(new FrontEnd(this, backend));
        }
        
    private:
        DISALLOW_COPY_AND_ASSIGN(TypeFilterImpl);
    };
    
    class CXXSyntheticChildren : public SyntheticChildren
    {
    public:
        typedef SyntheticChildrenFrontEnd* (*CreateFrontEndCallback) (CXXSyntheticChildren*, lldb::ValueObjectSP);
    protected:
        CreateFrontEndCallback m_create_callback;
        std::string m_description;
    public:
        CXXSyntheticChildren (const SyntheticChildren::Flags& flags,
                              const char* description,
                              CreateFrontEndCallback callback) :
        SyntheticChildren(flags),
        m_create_callback(callback),
        m_description(description ? description : "")
        {
        }
        
        bool
        IsScripted ()
        {
            return false;
        }
        
        std::string
        GetDescription ();
        
        virtual SyntheticChildrenFrontEnd::AutoPointer
        GetFrontEnd (ValueObject &backend)
        {
            return SyntheticChildrenFrontEnd::AutoPointer(m_create_callback(this, backend.GetSP()));
        }
        
    private:
        DISALLOW_COPY_AND_ASSIGN(CXXSyntheticChildren);
    };
    
#ifndef LLDB_DISABLE_PYTHON
    
    class ScriptedSyntheticChildren : public SyntheticChildren
    {
        std::string m_python_class;
        std::string m_python_code;
    public:
        
        ScriptedSyntheticChildren (const SyntheticChildren::Flags& flags,
                                   const char* pclass,
                                   const char* pcode = NULL) :
        SyntheticChildren(flags),
        m_python_class(),
        m_python_code()
        {
            if (pclass)
                m_python_class = pclass;
            if (pcode)
                m_python_code = pcode;
        }
        
        const char*
        GetPythonClassName ()
        {
            return m_python_class.c_str();
        }
        
        const char*
        GetPythonCode ()
        {
            return m_python_code.c_str();
        }
        
        void
        SetPythonClassName (const char* fname)
        {
            m_python_class.assign(fname);
            m_python_code.clear();
        }
        
        void
        SetPythonCode (const char* script)
        {
            m_python_code.assign(script);
        }
        
        std::string
        GetDescription ();
        
        bool
        IsScripted ()
        {
            return true;
        }
        
        class FrontEnd : public SyntheticChildrenFrontEnd
        {
        private:
            std::string m_python_class;
            lldb::ScriptInterpreterObjectSP m_wrapper_sp;
            ScriptInterpreter *m_interpreter;
        public:
            
            FrontEnd (std::string pclass,
                      ValueObject &backend);
            
            virtual
            ~FrontEnd ();
            
            virtual size_t
            CalculateNumChildren ()
            {
                if (!m_wrapper_sp || m_interpreter == NULL)
                    return 0;
                return m_interpreter->CalculateNumChildren(m_wrapper_sp);
            }
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update ()
            {
                if (!m_wrapper_sp || m_interpreter == NULL)
                    return false;
                
                return m_interpreter->UpdateSynthProviderInstance(m_wrapper_sp);
            }
            
            virtual bool
            MightHaveChildren ()
            {
                if (!m_wrapper_sp || m_interpreter == NULL)
                    return false;
                
                return m_interpreter->MightHaveChildrenSynthProviderInstance(m_wrapper_sp);
            }
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name)
            {
                if (!m_wrapper_sp || m_interpreter == NULL)
                    return UINT32_MAX;
                return m_interpreter->GetIndexOfChildWithName(m_wrapper_sp, name.GetCString());
            }
            
            typedef std::shared_ptr<SyntheticChildrenFrontEnd> SharedPointer;
            
        private:
            DISALLOW_COPY_AND_ASSIGN(FrontEnd);
        };
        
        virtual SyntheticChildrenFrontEnd::AutoPointer
        GetFrontEnd(ValueObject &backend)
        {
            return SyntheticChildrenFrontEnd::AutoPointer(new FrontEnd(m_python_class, backend));
        }    
        
    private:
        DISALLOW_COPY_AND_ASSIGN(ScriptedSyntheticChildren);
    };
#endif
} // namespace lldb_private

#endif	// lldb_TypeSynthetic_h_
