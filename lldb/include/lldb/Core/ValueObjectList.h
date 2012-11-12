//===-- ValueObjectList.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectList_h_
#define liblldb_ValueObjectList_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/UserID.h"
#include "lldb/Target/ExecutionContextScope.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A collection of ValueObject values that
//----------------------------------------------------------------------
class ValueObjectList
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ValueObjectList ();

    ValueObjectList (const ValueObjectList &rhs);

    ~ValueObjectList();

    const ValueObjectList &
    operator = (const ValueObjectList &rhs);

    void
    Append (const lldb::ValueObjectSP &val_obj_sp);

    void
    Append (const ValueObjectList &valobj_list);

    lldb::ValueObjectSP
    FindValueObjectByPointer (ValueObject *valobj);

    uint32_t
    GetSize () const;
    
    void
    Resize (uint32_t size);

    lldb::ValueObjectSP
    GetValueObjectAtIndex (uint32_t idx);

    lldb::ValueObjectSP
    RemoveValueObjectAtIndex (uint32_t idx);
    
    void
    SetValueObjectAtIndex (uint32_t idx, 
                           const lldb::ValueObjectSP &valobj_sp);
    
    lldb::ValueObjectSP
    FindValueObjectByValueName (const char *name);

    lldb::ValueObjectSP
    FindValueObjectByUID (lldb::user_id_t uid);

    void
    Swap (ValueObjectList &value_object_list);
    
protected:
    typedef std::vector<lldb::ValueObjectSP> collection;
    //------------------------------------------------------------------
    // Classes that inherit from ValueObjectList can see and modify these
    //------------------------------------------------------------------
    collection  m_value_objects;

};


} // namespace lldb_private

#endif  // liblldb_ValueObjectList_h_
