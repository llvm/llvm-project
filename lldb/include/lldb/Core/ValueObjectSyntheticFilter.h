//===-- ValueObjectSyntheticFilter.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectSyntheticFilter_h_
#define liblldb_ValueObjectSyntheticFilter_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A ValueObject that obtains its children from some source other than
// real information
// This is currently used to implement Python-based children and filters
// but you can bind it to any source of synthetic information and have
// it behave accordingly
//----------------------------------------------------------------------
class ValueObjectSynthetic : public ValueObject
{
public:
    virtual
    ~ValueObjectSynthetic();

    virtual size_t
    GetByteSize();
    
    virtual ConstString
    GetTypeName();

    virtual bool
    MightHaveChildren();

    virtual uint32_t
    CalculateNumChildren();

    virtual lldb::ValueType
    GetValueType() const;
    
    virtual lldb::ValueObjectSP
    GetChildAtIndex (uint32_t idx, bool can_create);
    
    virtual lldb::ValueObjectSP
    GetChildMemberWithName (const ConstString &name, bool can_create);
    
    virtual uint32_t
    GetIndexOfChildWithName (const ConstString &name);

    virtual bool
    IsInScope ();
    
    virtual bool
    HasSyntheticValue()
    {
        return false;
    }
    
    virtual bool
    IsSynthetic() { return true; }
    
    virtual void
    CalculateSyntheticValue (bool use_synthetic)
    {
    }
    
    virtual bool
    IsDynamic ()
    {
        if (m_parent)
            return m_parent->IsDynamic();
        else
            return false;
    }
    
    virtual ValueObject *
    GetParent()
    {
        if (m_parent)
            return m_parent->GetParent();
        else
            return NULL;
    }

    virtual const ValueObject *
    GetParent() const
    {
        if (m_parent)
            return m_parent->GetParent();
        else
            return NULL;
    }
    
    virtual lldb::ValueObjectSP
    GetNonSyntheticValue ();
    
    virtual bool
    ResolveValue (Scalar &scalar)
    {
        if (m_parent)
            return m_parent->ResolveValue(scalar);
        return false;
    }
    
protected:
    virtual bool
    UpdateValue ();
    
    virtual clang::ASTContext *
    GetClangASTImpl ();
    
    virtual lldb::clang_type_t
    GetClangTypeImpl ();
    
    virtual void
    CreateSynthFilter ();

    // we need to hold on to the SyntheticChildren because someone might delete the type binding while we are alive
    lldb::SyntheticChildrenSP m_synth_sp;
    std::auto_ptr<SyntheticChildrenFrontEnd> m_synth_filter_ap;
    
    typedef std::map<uint32_t, ValueObject*> ByIndexMap;
    typedef std::map<const char*, uint32_t> NameToIndexMap;
    
    typedef ByIndexMap::iterator ByIndexIterator;
    typedef NameToIndexMap::iterator NameToIndexIterator;

    ByIndexMap      m_children_byindex;
    NameToIndexMap  m_name_toindex;
    uint32_t        m_synthetic_children_count; // FIXME use the ValueObject's ChildrenManager instead of a special purpose solution
    
    ConstString     m_parent_type_name;

    LazyBool        m_might_have_children;
    
private:
    friend class ValueObject;
    ValueObjectSynthetic (ValueObject &parent, lldb::SyntheticChildrenSP filter);
    
    void
    CopyParentData ();
    
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectSynthetic);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectSyntheticFilter_h_
