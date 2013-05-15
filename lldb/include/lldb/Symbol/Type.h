//===-- Type.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Type_h_
#define liblldb_Type_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/UserID.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/Declaration.h"
#include <set>

namespace lldb_private {

class SymbolFileType :
    public std::enable_shared_from_this<SymbolFileType>,
    public UserID
    {
    public:
        SymbolFileType (SymbolFile &symbol_file, lldb::user_id_t uid) :
            UserID (uid),
            m_symbol_file (symbol_file)
        {
        }

        ~SymbolFileType ()
        {
        }

        Type *
        operator->()
        {
            return GetType ();
        }

        Type *
        GetType ();

    protected:
        SymbolFile &m_symbol_file;
        lldb::TypeSP m_type_sp;
    };
    
class Type :
    public std::enable_shared_from_this<Type>,
    public UserID
{
public:
    typedef enum EncodingDataTypeTag
    {
        eEncodingInvalid,
        eEncodingIsUID,                 ///< This type is the type whose UID is m_encoding_uid
        eEncodingIsConstUID,            ///< This type is the type whose UID is m_encoding_uid with the const qualifier added
        eEncodingIsRestrictUID,         ///< This type is the type whose UID is m_encoding_uid with the restrict qualifier added
        eEncodingIsVolatileUID,         ///< This type is the type whose UID is m_encoding_uid with the volatile qualifier added
        eEncodingIsTypedefUID,          ///< This type is pointer to a type whose UID is m_encoding_uid
        eEncodingIsPointerUID,          ///< This type is pointer to a type whose UID is m_encoding_uid
        eEncodingIsLValueReferenceUID,  ///< This type is L value reference to a type whose UID is m_encoding_uid
        eEncodingIsRValueReferenceUID,  ///< This type is R value reference to a type whose UID is m_encoding_uid
        eEncodingIsSyntheticUID
    } EncodingDataType;

    typedef enum ResolveStateTag
    {
        eResolveStateUnresolved = 0,
        eResolveStateForward    = 1,
        eResolveStateLayout     = 2,
        eResolveStateFull       = 3
    } ResolveState;

    Type (lldb::user_id_t uid,
          SymbolFile* symbol_file,
          const ConstString &name,
          uint64_t byte_size,
          SymbolContextScope *context,
          lldb::user_id_t encoding_uid,
          EncodingDataType encoding_uid_type,
          const Declaration& decl,
          lldb::clang_type_t clang_qual_type,
          ResolveState clang_type_resolve_state);
    
    // This makes an invalid type.  Used for functions that return a Type when they
    // get an error.
    Type();
    
    Type (const Type &rhs);

    const Type&
    operator= (const Type& rhs);

    void
    Dump(Stream *s, bool show_context);

    void
    DumpTypeName(Stream *s);


    void
    GetDescription (Stream *s, lldb::DescriptionLevel level, bool show_name);

    SymbolFile *
    GetSymbolFile()
    {
        return m_symbol_file;
    }
    const SymbolFile *
    GetSymbolFile() const
    {
        return m_symbol_file;
    }

    TypeList*
    GetTypeList();

    const ConstString&
    GetName();

    uint64_t
    GetByteSize();

    uint32_t
    GetNumChildren (bool omit_empty_base_classes);

    bool
    IsAggregateType ();

    bool
    IsValidType ()
    {
        return m_encoding_uid_type != eEncodingInvalid;
    }

    bool
    IsTypedef ()
    {
        return m_encoding_uid_type == eEncodingIsTypedefUID;
    }
    
    lldb::TypeSP
    GetTypedefType();

    const ConstString &
    GetName () const
    {
        return m_name;
    }

    ConstString
    GetQualifiedName ();

    void
    DumpValue(ExecutionContext *exe_ctx,
              Stream *s,
              const DataExtractor &data,
              uint32_t data_offset,
              bool show_type,
              bool show_summary,
              bool verbose,
              lldb::Format format = lldb::eFormatDefault);

    bool
    DumpValueInMemory(ExecutionContext *exe_ctx,
                      Stream *s,
                      lldb::addr_t address,
                      AddressType address_type,
                      bool show_types,
                      bool show_summary,
                      bool verbose);

    bool
    ReadFromMemory (ExecutionContext *exe_ctx,
                    lldb::addr_t address,
                    AddressType address_type,
                    DataExtractor &data);

    bool
    WriteToMemory (ExecutionContext *exe_ctx,
                   lldb::addr_t address,
                   AddressType address_type,
                   DataExtractor &data);

    bool
    GetIsDeclaration() const;

    void
    SetIsDeclaration(bool b);

    bool
    GetIsExternal() const;

    void
    SetIsExternal(bool b);

    lldb::Format
    GetFormat ();

    lldb::Encoding
    GetEncoding (uint64_t &count);

    SymbolContextScope *
    GetSymbolContextScope()
    {
        return m_context;
    }
    const SymbolContextScope *
    GetSymbolContextScope() const
    {
        return m_context;
    }
    void
    SetSymbolContextScope(SymbolContextScope *context)
    {
        m_context = context;
    }

    const lldb_private::Declaration &
    GetDeclaration () const;

    // Get the clang type, and resolve definitions for any 
    // class/struct/union/enum types completely.
    lldb::clang_type_t 
    GetClangFullType ();

    // Get the clang type, and resolve definitions enough so that the type could
    // have layout performed. This allows ptrs and refs to class/struct/union/enum 
    // types remain forward declarations.
    lldb::clang_type_t 
    GetClangLayoutType ();

    // Get the clang type and leave class/struct/union/enum types as forward
    // declarations if they haven't already been fully defined.
    lldb::clang_type_t 
    GetClangForwardType ();

    clang::ASTContext *
    GetClangAST ();

    ClangASTContext &
    GetClangASTContext ();

    static int
    Compare(const Type &a, const Type &b);

    // From a fully qualified typename, split the type into the type basename
    // and the remaining type scope (namespaces/classes).
    static bool
    GetTypeScopeAndBasename (const char* &name_cstr,
                             std::string &scope,
                             std::string &basename,
                             lldb::TypeClass &type_class);
    void
    SetEncodingType (Type *encoding_type)
    {
        m_encoding_type = encoding_type;
    }

    uint32_t
    GetEncodingMask ();

    void *
    CreateClangPointerType (Type *type);

    void *
    CreateClangTypedefType (Type *typedef_type, Type *base_type);

    // For C++98 references (&)
    void *
    CreateClangLValueReferenceType (Type *type);

    // For C++0x references (&&)
    void *
    CreateClangRValueReferenceType (Type *type);
    
    bool
    IsRealObjCClass();
    
    bool
    IsCompleteObjCClass()
    {
        return m_flags.is_complete_objc_class;
    }
    
    void
    SetIsCompleteObjCClass(bool is_complete_objc_class)
    {
        m_flags.is_complete_objc_class = is_complete_objc_class;
    }

protected:
    ConstString m_name;
    SymbolFile *m_symbol_file;
    SymbolContextScope *m_context; // The symbol context in which this type is defined
    Type *m_encoding_type;
    lldb::user_id_t m_encoding_uid;
    EncodingDataType m_encoding_uid_type;
    uint64_t m_byte_size;
    Declaration m_decl;
    lldb::clang_type_t m_clang_type;
    
    struct Flags {
        ResolveState    clang_type_resolve_state : 2;
        bool            is_complete_objc_class   : 1;
    } m_flags;

    Type *
    GetEncodingType ();
    
    bool 
    ResolveClangType (ResolveState clang_type_resolve_state);
};


///
/// Sometimes you can find the name of the type corresponding to an object, but we don't have debug
/// information for it.  If that is the case, you can return one of these objects, and then if it
/// has a full type, you can use that, but if not at least you can print the name for informational
/// purposes.
///

class TypeAndOrName
{
public:
    TypeAndOrName ();
    TypeAndOrName (lldb::TypeSP &type_sp);
    TypeAndOrName (const char *type_str);
    TypeAndOrName (const TypeAndOrName &rhs);
    TypeAndOrName (ConstString &type_const_string);
    
    TypeAndOrName &
    operator= (const TypeAndOrName &rhs);
    
    bool
    operator==(const TypeAndOrName &other) const;
    
    bool
    operator!=(const TypeAndOrName &other) const;
    
    ConstString GetName () const;

    lldb::TypeSP
    GetTypeSP () const 
    {
        return m_type_sp;
    }
    
    void
    SetName (const ConstString &type_name);
    
    void 
    SetName (const char *type_name_cstr);
    
    void
    SetTypeSP (lldb::TypeSP type_sp);
    
    bool
    IsEmpty ();
    
    bool
    HasName ();
    
    bool
    HasTypeSP ();
    
    void
    Clear ();
    
    operator
    bool ()
    {
        return !IsEmpty();
    }
    
private:
    lldb::TypeSP m_type_sp;
    ConstString m_type_name;
};

// the two classes here are used by the public API as a backend to
// the SBType and SBTypeList classes
    
class TypeImpl
{
public:
    
    TypeImpl() :
        m_clang_ast_type(),
        m_type_sp()
    {
    }
    
    TypeImpl(const TypeImpl& rhs) :
        m_clang_ast_type(rhs.m_clang_ast_type),
        m_type_sp(rhs.m_type_sp)
    {
    }
    
    TypeImpl(const lldb_private::ClangASTType& type);
    
    TypeImpl(const lldb::TypeSP& type);
    
    TypeImpl&
    operator = (const TypeImpl& rhs);
    
    bool
    operator == (const TypeImpl& rhs)
    {
        return m_clang_ast_type == rhs.m_clang_ast_type && m_type_sp.get() == rhs.m_type_sp.get();
    }

    bool
    operator != (const TypeImpl& rhs)
    {
        return m_clang_ast_type != rhs.m_clang_ast_type || m_type_sp.get() != rhs.m_type_sp.get();
    }
    
    bool
    IsValid()
    {
        return m_type_sp.get() != NULL || m_clang_ast_type.IsValid();
    }
    
    const lldb_private::ClangASTType &
    GetClangASTType() const
    {
        return m_clang_ast_type;
    }
    
    clang::ASTContext*
    GetASTContext();
    
    lldb::clang_type_t
    GetOpaqueQualType();    

    lldb::TypeSP
    GetTypeSP ()
    {
        return m_type_sp;
    }

    bool
    GetDescription (lldb_private::Stream &strm, 
                    lldb::DescriptionLevel description_level);
    
    void
    SetType (const lldb::TypeSP &type_sp);

private:
    ClangASTType m_clang_ast_type;
    lldb::TypeSP m_type_sp;
};

class TypeListImpl
{
public:
    TypeListImpl() :
        m_content() 
    {
    }
    
    void
    Append (const lldb::TypeImplSP& type)
    {
        m_content.push_back(type);
    }
    
    lldb::TypeImplSP
    GetTypeAtIndex(size_t idx)
    {
        lldb::TypeImplSP type_sp;
        if (idx < GetSize())
            type_sp = m_content[idx];
        return type_sp;
    }
    
    size_t
    GetSize()
    {
        return m_content.size();
    }
    
private:
    std::vector<lldb::TypeImplSP> m_content;
};
    
class TypeMemberImpl
{
public:
    TypeMemberImpl () :
        m_type_impl_sp (),
        m_bit_offset (0),
        m_name (),
        m_bitfield_bit_size (0),
        m_is_bitfield (false)

    {
    }

    TypeMemberImpl (const lldb::TypeImplSP &type_impl_sp, 
                    uint64_t bit_offset,
                    const ConstString &name,
                    uint32_t bitfield_bit_size = 0,
                    bool is_bitfield = false) :
        m_type_impl_sp (type_impl_sp),
        m_bit_offset (bit_offset),
        m_name (name),
        m_bitfield_bit_size (bitfield_bit_size),
        m_is_bitfield (is_bitfield)
    {
    }
    
    TypeMemberImpl (const lldb::TypeImplSP &type_impl_sp, 
                    uint64_t bit_offset):
        m_type_impl_sp (type_impl_sp),
        m_bit_offset (bit_offset),
        m_name (),
        m_bitfield_bit_size (0),
        m_is_bitfield (false)
    {
    }

    const lldb::TypeImplSP &
    GetTypeImpl ()
    {
        return m_type_impl_sp;
    }

    const ConstString &
    GetName () const
    {
        return m_name;
    }
    
    uint64_t
    GetBitOffset () const
    {
        return m_bit_offset;
    }

    uint32_t
    GetBitfieldBitSize () const
    {
        return m_bitfield_bit_size;
    }

    void
    SetBitfieldBitSize (uint32_t bitfield_bit_size)
    {
        m_bitfield_bit_size = bitfield_bit_size;
    }

    bool
    GetIsBitfield () const
    {
        return m_is_bitfield;
    }
    
    void
    SetIsBitfield (bool is_bitfield)
    {
        m_is_bitfield = is_bitfield;
    }

protected:
    lldb::TypeImplSP m_type_impl_sp;
    uint64_t m_bit_offset;
    ConstString m_name;
    uint32_t m_bitfield_bit_size; // Bit size for bitfield members only
    bool m_is_bitfield;
};

    
} // namespace lldb_private

#endif  // liblldb_Type_h_

