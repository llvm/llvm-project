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
#include "lldb/Symbol/Type.h"

namespace lldb_private {

class TypeFormatImpl
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
    
    TypeFormatImpl (lldb::Format f = lldb::eFormatInvalid,
                 const Flags& flags = Flags());
    
    typedef STD_SHARED_PTR(TypeFormatImpl) SharedPointer;
    typedef bool(*ValueCallback)(void*, ConstString, const lldb::TypeFormatImplSP&);
    
    ~TypeFormatImpl ()
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
    
    lldb::Format
    GetFormat () const
    {
        return m_format;
    }
    
    void
    SetFormat (lldb::Format fmt)
    {
        m_format = fmt;
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
    
    uint32_t&
    GetRevision ()
    {
        return m_my_revision;
    }
    
    std::string
    GetDescription();
    
protected:
    Flags m_flags;
    lldb::Format m_format;
    uint32_t m_my_revision;
    
private:
    DISALLOW_COPY_AND_ASSIGN(TypeFormatImpl);
};

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
    
    virtual uint32_t
    CalculateNumChildren () = 0;
    
    virtual lldb::ValueObjectSP
    GetChildAtIndex (uint32_t idx) = 0;
    
    virtual uint32_t
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
    
    typedef STD_SHARED_PTR(SyntheticChildrenFrontEnd) SharedPointer;
    typedef std::auto_ptr<SyntheticChildrenFrontEnd> AutoPointer;
    
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
    
    typedef STD_SHARED_PTR(SyntheticChildren) SharedPointer;
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
    AddExpressionPath (std::string path)
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
    SetExpressionPathAtIndex (int i, std::string path)
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
    IsScripted()
    {
        return false;
    }
    
    std::string
    GetDescription();
    
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
        ~FrontEnd()
        {
        }
        
        virtual uint32_t
        CalculateNumChildren()
        {
            return filter->GetCount();
        }
        
        virtual lldb::ValueObjectSP
        GetChildAtIndex (uint32_t idx)
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
        
        virtual uint32_t
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
        
        typedef STD_SHARED_PTR(SyntheticChildrenFrontEnd) SharedPointer;
        
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
        CXXSyntheticChildren(const SyntheticChildren::Flags& flags,
                             const char* description,
                             CreateFrontEndCallback callback) :
        SyntheticChildren(flags),
        m_create_callback(callback),
        m_description(description ? description : "")
        {
        }
        
        bool
        IsScripted()
        {
            return false;
        }
        
        std::string
        GetDescription();
                
        virtual SyntheticChildrenFrontEnd::AutoPointer
        GetFrontEnd(ValueObject &backend)
        {
            return SyntheticChildrenFrontEnd::AutoPointer(m_create_callback(this, backend.GetSP()));
        }
        
    private:
        DISALLOW_COPY_AND_ASSIGN(CXXSyntheticChildren);
    };

#ifndef LLDB_DISABLE_PYTHON

class TypeSyntheticImpl : public SyntheticChildren
{
    std::string m_python_class;
    std::string m_python_code;
public:
    
    TypeSyntheticImpl(const SyntheticChildren::Flags& flags,
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
    GetPythonClassName()
    {
        return m_python_class.c_str();
    }

    const char*
    GetPythonCode()
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
    GetDescription();
    
    bool
    IsScripted()
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
        
        FrontEnd(std::string pclass,
                 ValueObject &backend);
        
        virtual
        ~FrontEnd();
        
        virtual uint32_t
        CalculateNumChildren()
        {
            if (!m_wrapper_sp || m_interpreter == NULL)
                return 0;
            return m_interpreter->CalculateNumChildren(m_wrapper_sp);
        }
        
        virtual lldb::ValueObjectSP
        GetChildAtIndex (uint32_t idx);
        
        virtual bool
        Update()
        {
            if (!m_wrapper_sp || m_interpreter == NULL)
                return false;
            
            return m_interpreter->UpdateSynthProviderInstance(m_wrapper_sp);
        }
        
        virtual bool
        MightHaveChildren()
        {
            if (!m_wrapper_sp || m_interpreter == NULL)
                return false;
            
            return m_interpreter->MightHaveChildrenSynthProviderInstance(m_wrapper_sp);
        }
        
        virtual uint32_t
        GetIndexOfChildWithName (const ConstString &name)
        {
            if (!m_wrapper_sp || m_interpreter == NULL)
                return UINT32_MAX;
            return m_interpreter->GetIndexOfChildWithName(m_wrapper_sp, name.GetCString());
        }
        
        typedef STD_SHARED_PTR(SyntheticChildrenFrontEnd) SharedPointer;

    private:
        DISALLOW_COPY_AND_ASSIGN(FrontEnd);
    };
    
    virtual SyntheticChildrenFrontEnd::AutoPointer
    GetFrontEnd(ValueObject &backend)
    {
        return SyntheticChildrenFrontEnd::AutoPointer(new FrontEnd(m_python_class, backend));
    }    
    
private:
    DISALLOW_COPY_AND_ASSIGN(TypeSyntheticImpl);
};

#endif // #ifndef LLDB_DISABLE_PYTHON
class SyntheticArrayView : public SyntheticChildren
{
public:
    
    struct SyntheticArrayRange
    {
    private:
        int m_low;
        int m_high;
        SyntheticArrayRange* m_next;
        
    public:
        
        SyntheticArrayRange () : 
        m_low(-1),
        m_high(-2),
        m_next(NULL)
        {}
        
        SyntheticArrayRange (int L) : 
        m_low(L),
        m_high(L),
        m_next(NULL)
        {}
        
        SyntheticArrayRange (int L, int H) : 
        m_low(L),
        m_high(H),
        m_next(NULL)
        {}
        
        SyntheticArrayRange (int L, int H, SyntheticArrayRange* N) : 
        m_low(L),
        m_high(H),
        m_next(N)
        {}
        
        inline int
        GetLow ()
        {
            return m_low;
        }
        
        inline int
        GetHigh ()
        {
            return m_high;
        }
        
        inline void
        SetLow (int L)
        {
            m_low = L;
        }
        
        inline void
        SetHigh (int H)
        {
            m_high = H;
        }
        
        inline  int
        GetSelfCount()
        {
            return GetHigh() - GetLow() + 1;
        }
        
        int
        GetCount()
        {
            int count = GetSelfCount();
            if (m_next)
                count += m_next->GetCount();
            return count;
        }
        
        inline SyntheticArrayRange*
        GetNext()
        {
            return m_next;
        }
        
        void
        SetNext(SyntheticArrayRange* N)
        {
            if (m_next)
                delete m_next;
            m_next = N;
        }
        
        void
        SetNext(int L, int H)
        {
            if (m_next)
                delete m_next;
            m_next = new SyntheticArrayRange(L, H);
        }
        
        void
        SetNext(int L)
        {
            if (m_next)
                delete m_next;
            m_next = new SyntheticArrayRange(L);
        }
        
        ~SyntheticArrayRange()
        {
            delete m_next;
            m_next = NULL;
        }
        
    };
    
    SyntheticArrayView(const SyntheticChildren::Flags& flags) :
        SyntheticChildren(flags),
        m_head(),
        m_tail(&m_head)
    {
    }
    
    void
    AddRange(int L, int H)
    {
        m_tail->SetLow(L);
        m_tail->SetHigh(H);
        m_tail->SetNext(new SyntheticArrayRange());
        m_tail = m_tail->GetNext();
    }
    
    int
    GetCount()
    {
        return m_head.GetCount();
    }
    
    int
    GetRealIndexForIndex(int i);
    
    bool
    IsScripted()
    {
        return false;
    }
    
    std::string
    GetDescription();
    
    class FrontEnd : public SyntheticChildrenFrontEnd
    {
    private:
        SyntheticArrayView* filter;
    public:
        
        FrontEnd(SyntheticArrayView* flt,
                 ValueObject &backend) :
        SyntheticChildrenFrontEnd(backend),
        filter(flt)
        {}
        
        virtual
        ~FrontEnd()
        {
        }
        
        virtual uint32_t
        CalculateNumChildren()
        {
            return filter->GetCount();
        }
        
        virtual bool
        MightHaveChildren ()
        {
            return filter->GetCount() > 0;
        }
        
        virtual lldb::ValueObjectSP
        GetChildAtIndex (uint32_t idx)
        {
            if (idx >= filter->GetCount())
                return lldb::ValueObjectSP();
            return m_backend.GetSyntheticArrayMember(filter->GetRealIndexForIndex(idx), true);
        }
        
        virtual bool
        Update() { return false; }
        
        virtual uint32_t
        GetIndexOfChildWithName (const ConstString &name_cs);
        
        typedef STD_SHARED_PTR(SyntheticChildrenFrontEnd) SharedPointer;
    
    private:
        DISALLOW_COPY_AND_ASSIGN(FrontEnd);
    };
    
    virtual SyntheticChildrenFrontEnd::AutoPointer
    GetFrontEnd(ValueObject &backend)
    {
        return SyntheticChildrenFrontEnd::AutoPointer(new FrontEnd(this, backend));
    }
private:
    SyntheticArrayRange m_head;
    SyntheticArrayRange *m_tail;

private:
    DISALLOW_COPY_AND_ASSIGN(SyntheticArrayView);
};


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
    IsScripted() = 0;
    
    virtual Type
    GetType () = 0;
    
    uint32_t&
    GetRevision ()
    {
        return m_my_revision;
    }
    
    typedef STD_SHARED_PTR(TypeSummaryImpl) SharedPointer;
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
    IsScripted()
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
    typedef bool (*Callback)(ValueObject& valobj,
                             Stream& dest);
    
    
    Callback m_impl;
    std::string m_description;
    
    CXXFunctionSummaryFormat(const TypeSummaryImpl::Flags& flags,
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
    ~CXXFunctionSummaryFormat()
    {
    }
    
    virtual bool
    FormatObject(ValueObject *valobj,
                 std::string& dest);
    
    virtual std::string
    GetDescription();
    
    virtual bool
    IsScripted()
    {
        return false;
    }
    
    virtual Type
    GetType ()
    {
        return TypeSummaryImpl::eTypeCallback;
    }
    
    typedef STD_SHARED_PTR(CXXFunctionSummaryFormat) SharedPointer;

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
    ~ScriptSummaryFormat()
    {
    }
    
    virtual bool
    FormatObject(ValueObject *valobj,
                 std::string& dest);
    
    virtual std::string
    GetDescription();
    
    virtual bool
    IsScripted()
    {
        return true;
    }
    
    virtual Type
    GetType ()
    {
        return TypeSummaryImpl::eTypeScript;
    }
    
    typedef STD_SHARED_PTR(ScriptSummaryFormat) SharedPointer;

    
private:
    DISALLOW_COPY_AND_ASSIGN(ScriptSummaryFormat);
};

#endif // #ifndef LLDB_DISABLE_PYTHON

// TODO: at the moment, this class is only used as a backing store for SBTypeNameSpecifier in the public API
// In the future, this might be used as the basic unit for typename-to-formatter matching, replacing
// the current plain/regexp distinction in FormatNavigator<>
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
