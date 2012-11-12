//===-- TypeHierarchyNavigator.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_TypeHierarchyNavigator_h_
#define lldb_TypeHierarchyNavigator_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclObjC.h"

// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

namespace lldb_private {

class TypeHierarchyNavigator {

public:
    
    enum RelationshipToCurrentType
    {
        eRootType,
        eCXXBaseClass,
        eCXXVBaseClass,
        eObjCBaseClass,
        eStrippedPointer,
        eStrippedReference,
        eStrippedTypedef
    };
    
    typedef bool (*TypeHierarchyNavigatorCallback)(const clang::QualType& qual_type,
                                                   RelationshipToCurrentType reason_why_here,
                                                   void* callback_baton);
    
    TypeHierarchyNavigator(const clang::QualType& qual_type,
                           ValueObject& val_obj,
                           void* callback_baton = NULL) : 
        m_root_type(qual_type),
        m_value_object(val_obj),
        m_default_callback_baton(callback_baton)
    {
    }
        
    bool
    LoopThrough(TypeHierarchyNavigatorCallback callback,
                void* callback_baton = NULL);
    
private:
    
    bool
    LoopThrough(const clang::QualType& qual_type,
                TypeHierarchyNavigatorCallback callback,
                RelationshipToCurrentType reason_why_here,
                void* callback_baton);
    
    const clang::QualType& m_root_type;
    ValueObject& m_value_object;
    void* m_default_callback_baton;
    
};
    
} // namespace lldb_private

#endif  // lldb_TypeHierarchyNavigator_h_
