//===-- SBValueList.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBValueList_h_
#define LLDB_SBValueList_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBValueList
{
public:

    SBValueList ();

    SBValueList (const lldb::SBValueList &rhs);

    ~SBValueList();

    bool
    IsValid() const;
    
    void
    Clear();

    void
    Append (const lldb::SBValue &val_obj);

    void
    Append (const lldb::SBValueList& value_list);

    uint32_t
    GetSize() const;

    lldb::SBValue
    GetValueAtIndex (uint32_t idx) const;

    lldb::SBValue
    FindValueObjectByUID (lldb::user_id_t uid);

    const lldb::SBValueList &
    operator = (const lldb::SBValueList &rhs);

    lldb_private::ValueObjectList *
    operator -> ();

    lldb_private::ValueObjectList &
    operator* ();

    const lldb_private::ValueObjectList *
    operator -> () const;

    const lldb_private::ValueObjectList &
    operator* () const;
    
    lldb_private::ValueObjectList *
    get ();

    lldb_private::ValueObjectList &
    ref ();

private:
    friend class SBFrame;

    SBValueList (const lldb_private::ValueObjectList *lldb_object_ptr);

    void
    Append (lldb::ValueObjectSP& val_obj_sp);

    void
    CreateIfNeeded ();

    std::auto_ptr<lldb_private::ValueObjectList> m_opaque_ap;
};


} // namespace lldb

#endif  // LLDB_SBValueList_h_
