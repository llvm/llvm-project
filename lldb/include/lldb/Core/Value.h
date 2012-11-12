//===-- Value.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Value_h_
#define liblldb_Value_h_

// C Includes
// C++ Includes
#include <string>
#include <vector>
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Scalar.h"

namespace lldb_private {

class Value
{
public:

    // Values Less than zero are an error, greater than or equal to zero
    // returns what the Scalar result is.
    enum ValueType
    {
                                        // m_value contains...
                                        // ============================
        eValueTypeScalar,               // raw scalar value
        eValueTypeVector,               // byte array of m_vector.length with endianness of m_vector.byte_order
        eValueTypeFileAddress,          // file address value
        eValueTypeLoadAddress,          // load address value
        eValueTypeHostAddress           // host address value (for memory in the process that is using liblldb)
    };

    enum ContextType                    // Type that describes Value::m_context
    {
                                        // m_context contains...
                                        // ====================
        eContextTypeInvalid,            // undefined
        eContextTypeClangType,          // void * (an opaque clang::QualType * that can be fed to "static QualType QualType::getFromOpaquePtr(void *)")
        eContextTypeRegisterInfo,       // RegisterInfo * (can be a scalar or a vector register)
        eContextTypeLLDBType,           // lldb_private::Type *
        eContextTypeVariable            // lldb_private::Variable *
    };

    enum
    {
        kMaxByteSize = 32u
    };

    struct Vector
    {
        // The byte array must be big enough to hold vector registers for any supported target.
        uint8_t bytes[kMaxByteSize];
        size_t length;
        lldb::ByteOrder byte_order;

        Vector() : 
			length(0), 
			byte_order(lldb::eByteOrderInvalid) 
        {
		}

        Vector(const Vector& vector) 
		{ *this = vector; 
        }
        const Vector& 
		operator=(const Vector& vector) 
		{
            SetBytes((uint8_t *)vector.bytes, vector.length, vector.byte_order);
            return *this;
        }

        bool 
		SetBytes(uint8_t *bytes, size_t length, lldb::ByteOrder byte_order)
		{
            this->length = length;
            this->byte_order = byte_order;
            if (length)
                ::memcpy(this->bytes, bytes, length < kMaxByteSize ? length : kMaxByteSize);
            return IsValid();
        }

        bool
		IsValid() const 
		{
            return (length > 0 && length < kMaxByteSize && byte_order != lldb::eByteOrderInvalid);
        }
        // Casts a vector, if valid, to an unsigned int of matching or largest supported size.
        // Truncates to the beginning of the vector if required.
        // Returns a default constructed Scalar if the Vector data is internally inconsistent.
        Scalar 
		GetAsScalar() const 
		{
            Scalar scalar;
            if (IsValid())
                if (length == 1) scalar = *(uint8_t *)bytes;
                if (length == 2) scalar = *(uint16_t *)bytes;
                if (length == 4) scalar = *(uint32_t *)bytes;
                if (length == 8) scalar = *(uint64_t *)bytes;
#if defined (ENABLE_128_BIT_SUPPORT)
                if (length >= 16) scalar = *(__uint128_t *)bytes;
#else
                if (length >= 16) scalar = *(__uint64_t *)bytes;
#endif
            return scalar;
        }
    };

    Value();
    Value(const Scalar& scalar);
    Value(const Vector& vector);
    Value(const uint8_t *bytes, int len);
    Value(const Value &rhs);
    
    Value &
    operator=(const Value &rhs);

    lldb::clang_type_t
    GetClangType();

    ValueType
    GetValueType() const;

    AddressType
    GetValueAddressType () const;

    ContextType
    GetContextType() const
    {
        return m_context_type;
    }

    void
    SetValueType (ValueType value_type)
    {
        m_value_type = value_type;
    }

    void
    ClearContext ()
    {
        m_context = NULL;
        m_context_type = eContextTypeInvalid;
    }

    void
    SetContext (ContextType context_type, void *p)
    {
        m_context_type = context_type;
        m_context = p;
        if (m_context_type == eContextTypeRegisterInfo) {
            RegisterInfo *reg_info = GetRegisterInfo();
            if (reg_info->encoding == lldb::eEncodingVector)
                SetValueType(eValueTypeVector);
            else
                SetValueType(eValueTypeScalar);
        }
    }

    RegisterInfo *
    GetRegisterInfo();

    Type *
    GetType();

    Scalar &
    ResolveValue (ExecutionContext *exe_ctx, clang::ASTContext *ast_context);

    Scalar &
    GetScalar()
    {
        return m_value;
    }
    
    Vector &
    GetVector()
    {
        return m_vector;
    }

    bool
    SetVectorBytes(const Vector& vector) 
	{
        m_vector = vector;
        return m_vector.IsValid();
    }
    
    bool
    SetVectorBytes(uint8_t *bytes, size_t length, lldb::ByteOrder byte_order) 
	{
        return m_vector.SetBytes(bytes, length, byte_order);
    }

    bool
    SetScalarFromVector() 
	{
        if (m_vector.IsValid()) 
		{
            m_value = m_vector.GetAsScalar();
            return true;
        }
        return false;
    }

    void
    ResizeData(int len);

    bool
    ValueOf(ExecutionContext *exe_ctx, clang::ASTContext *ast_context);

    Variable *
    GetVariable();

    void
    Dump (Stream* strm);

    lldb::Format
    GetValueDefaultFormat ();

    size_t
    GetValueByteSize (clang::ASTContext *ast_context, Error *error_ptr);

    Error
    GetValueAsData (ExecutionContext *exe_ctx, 
                    clang::ASTContext *ast_context, 
                    DataExtractor &data, 
                    uint32_t data_offset,
                    Module *module);     // Can be NULL

    static const char *
    GetValueTypeAsCString (ValueType context_type);

    static const char *
    GetContextTypeAsCString (ContextType context_type);

    bool
    GetData (DataExtractor &data);

protected:
    Scalar          m_value;
    Vector          m_vector;
    ValueType       m_value_type;
    void *          m_context;
    ContextType     m_context_type;
    DataBufferHeap  m_data_buffer;
};

class ValueList
{
public:
    ValueList () :
        m_values()
    {
    }

    ValueList (const ValueList &rhs);

    ~ValueList () 
    {
    }

    const ValueList & operator= (const ValueList &rhs);

    // void InsertValue (Value *value, size_t idx);
    void PushValue (const Value &value);

    size_t GetSize ();
    Value *GetValueAtIndex(size_t idx);
    void Clear();

protected:

private:
    typedef std::vector<Value> collection;

    collection m_values;
};

} // namespace lldb_private

#endif  // liblldb_Value_h_
