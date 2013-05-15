//===-- ClangExternalASTSourceCommon.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExternalASTSourceCommon_h
#define liblldb_ClangExternalASTSourceCommon_h

// Clang headers like to use NDEBUG inside of them to enable/disable debug 
// releated features using "#ifndef NDEBUG" preprocessor blocks to do one thing
// or another. This is bad because it means that if clang was built in release
// mode, it assumes that you are building in release mode which is not always
// the case. You can end up with functions that are defined as empty in header
// files when NDEBUG is not defined, and this can cause link errors with the
// clang .a files that you have since you might be missing functions in the .a
// file. So we have to define NDEBUG when including clang headers to avoid any
// mismatches. This is covered by rdar://problem/8691220

#if !defined(NDEBUG) && !defined(LLVM_NDEBUG_OFF)
#define LLDB_DEFINED_NDEBUG_FOR_CLANG
#define NDEBUG
// Need to include assert.h so it is as clang would expect it to be (disabled)
#include <assert.h>
#endif

#include "clang/AST/ExternalASTSource.h"

#ifdef LLDB_DEFINED_NDEBUG_FOR_CLANG
#undef NDEBUG
#undef LLDB_DEFINED_NDEBUG_FOR_CLANG
// Need to re-include assert.h so it is as _we_ would expect it to be (enabled)
#include <assert.h>
#endif

#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/dwarf.h"

namespace lldb_private {

class ClangASTMetadata
{
public:
    ClangASTMetadata () :
        m_user_id(0),
        m_union_is_user_id(false),
        m_union_is_isa_ptr(false),
        m_has_object_ptr(false),
        m_is_self (false),
        m_is_dynamic_cxx (true)
    {
    }

    bool
    GetIsDynamicCXXType () const
    {
        return m_is_dynamic_cxx;
    }

    void
    SetIsDynamicCXXType (bool b)
    {
        m_is_dynamic_cxx = b;
    }

    void
    SetUserID (lldb::user_id_t user_id)
    {
        m_user_id = user_id;
        m_union_is_user_id = true;
        m_union_is_isa_ptr = false;
    }

    lldb::user_id_t
    GetUserID () const
    {
        if (m_union_is_user_id)
            return m_user_id;
        else
            return LLDB_INVALID_UID;
    }

    void
    SetISAPtr (uint64_t isa_ptr)
    {
        m_isa_ptr = isa_ptr;
        m_union_is_user_id = false;
        m_union_is_isa_ptr = true;
    }
    
    uint64_t
    GetISAPtr () const
    {
        if (m_union_is_isa_ptr)
            return m_isa_ptr;
        else
            return 0;
    }
    
    void
    SetObjectPtrName(const char *name)
    {
        m_has_object_ptr = true;
        if (strcmp (name, "self") == 0)
            m_is_self = true;
        else if (strcmp (name, "this") == 0)
            m_is_self = false;
        else
            m_has_object_ptr = false;
    }
    
    lldb::LanguageType
    GetObjectPtrLanguage () const
    {
        if (m_has_object_ptr)
        {
            if (m_is_self)
                return lldb::eLanguageTypeObjC;
            else
                return lldb::eLanguageTypeC_plus_plus;
        }
        return lldb::eLanguageTypeUnknown;
            
    }
    const char *
    GetObjectPtrName() const
    {
        if (m_has_object_ptr)
        {
            if (m_is_self)
                return "self";
            else
                return "this";
        }
        else
            return NULL;
    }
    
    bool
    HasObjectPtr() const
    {
        return m_has_object_ptr;
    }
    
    void
    Dump (Stream *s);
    
private:
    union
    {
        lldb::user_id_t m_user_id;
        uint64_t  m_isa_ptr;
    };
    bool m_union_is_user_id : 1,
         m_union_is_isa_ptr : 1,
         m_has_object_ptr : 1,
         m_is_self : 1,
         m_is_dynamic_cxx : 1;
    
};

class ClangExternalASTSourceCommon : public clang::ExternalASTSource 
{
public:
    ClangExternalASTSourceCommon();
    ~ClangExternalASTSourceCommon();

    virtual ClangASTMetadata *GetMetadata(const void *object);
    virtual void SetMetadata(const void *object, ClangASTMetadata &metadata);
    virtual bool HasMetadata(const void *object);
private:
    typedef llvm::DenseMap<const void *, ClangASTMetadata> MetadataMap;
    
    MetadataMap m_metadata;
    uint64_t    m_magic;        ///< Because we don't have RTTI, we must take it
                                ///< on faith that any valid ExternalASTSource that
                                ///< we try to use the *Metadata APIs on inherits
                                ///< from ClangExternalASTSourceCommon.  This magic
                                ///< number exists to enforce that.
};

}

#endif
