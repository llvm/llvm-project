//===-- RegisterValue.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_RegisterValue_h
#define lldb_RegisterValue_h

// C Includes
#include <string.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-private.h"
#include "lldb/Host/Endian.h"

//#define ENABLE_128_BIT_SUPPORT 1
namespace lldb_private {

    class RegisterValue
    {
    public:
        enum
        {
            kMaxRegisterByteSize = 32u
        };
        enum Type
        {
            eTypeInvalid,
            eTypeUInt8,
            eTypeUInt16,
            eTypeUInt32,
            eTypeUInt64,
#if defined (ENABLE_128_BIT_SUPPORT)
            eTypeUInt128,
#endif
            eTypeFloat,
            eTypeDouble,
            eTypeLongDouble,
            eTypeBytes
        };
        
        RegisterValue () : 
            m_type (eTypeInvalid)
        {
        }

        explicit 
        RegisterValue (uint8_t inst) : 
            m_type (eTypeUInt8)
        {
            m_data.uint8 = inst;
        }

        explicit 
        RegisterValue (uint16_t inst) : 
            m_type (eTypeUInt16)
        {
            m_data.uint16 = inst;
        }

        explicit 
        RegisterValue (uint32_t inst) : 
            m_type (eTypeUInt32)
        {
            m_data.uint32 = inst;
        }

        explicit 
        RegisterValue (uint64_t inst) : 
            m_type (eTypeUInt64)
        {
            m_data.uint64 = inst;
        }

#if defined (ENABLE_128_BIT_SUPPORT)
        explicit 
        RegisterValue (__uint128_t inst) : 
            m_type (eTypeUInt128)
        {
            m_data.uint128 = inst;
        }
#endif        
        explicit 
        RegisterValue (float value) : 
            m_type (eTypeFloat)
        {
            m_data.ieee_float = value;
        }

        explicit 
        RegisterValue (double value) : 
            m_type (eTypeDouble)
        {
            m_data.ieee_double = value;
        }

        explicit 
        RegisterValue (long double value) : 
            m_type (eTypeLongDouble)
        {
            m_data.ieee_long_double = value;
        }

        explicit 
        RegisterValue (uint8_t *bytes, size_t length, lldb::ByteOrder byte_order)
        {
            SetBytes (bytes, length, byte_order);
        }

        RegisterValue::Type
        GetType () const
        {
            return m_type;
        }

        bool
        CopyValue (const RegisterValue &rhs);

        void
        SetType (RegisterValue::Type type)
        {
            m_type = type;
        }

        RegisterValue::Type
        SetType (const RegisterInfo *reg_info);
        
        bool
        GetData (DataExtractor &data) const;
        
        // Copy the register value from this object into a buffer in "dst"
        // and obey the "dst_byte_order" when copying the data. Also watch out
        // in case "dst_len" is longer or shorter than the register value 
        // described by "reg_info" and only copy the least significant bytes 
        // of the register value, or pad the destination with zeroes if the
        // register byte size is shorter that "dst_len" (all while correctly
        // abiding the "dst_byte_order"). Returns the number of bytes copied
        // into "dst".
        uint32_t
        GetAsMemoryData (const RegisterInfo *reg_info,
                         void *dst, 
                         uint32_t dst_len, 
                         lldb::ByteOrder dst_byte_order,
                         Error &error) const;

        uint32_t
        SetFromMemoryData (const RegisterInfo *reg_info,
                           const void *src,
                           uint32_t src_len,
                           lldb::ByteOrder src_byte_order,
                           Error &error);
                           
        bool
        GetScalarValue (Scalar &scalar) const;

        uint8_t
        GetAsUInt8 (uint8_t fail_value = UINT8_MAX, bool *success_ptr = NULL) const
        {
            if (m_type == eTypeUInt8)
            {
                if (success_ptr)
                    *success_ptr = true;                
                return m_data.uint8;
            }
            if (success_ptr)
                *success_ptr = true;
            return fail_value;
        }

        uint16_t
        GetAsUInt16 (uint16_t fail_value = UINT16_MAX, bool *success_ptr = NULL) const;

        uint32_t
        GetAsUInt32 (uint32_t fail_value = UINT32_MAX, bool *success_ptr = NULL) const;

        uint64_t
        GetAsUInt64 (uint64_t fail_value = UINT64_MAX, bool *success_ptr = NULL) const;

#if defined (ENABLE_128_BIT_SUPPORT)
        __uint128_t
        GetAsUInt128 (__uint128_t fail_value = ~((__uint128_t)0), bool *success_ptr = NULL) const;
#endif

        float
        GetAsFloat (float fail_value = 0.0f, bool *success_ptr = NULL) const;

        double
        GetAsDouble (double fail_value = 0.0, bool *success_ptr = NULL) const;

        long double
        GetAsLongDouble (long double fail_value = 0.0, bool *success_ptr = NULL) const;

        void
        SetValueToInvalid ()
        {
            m_type = eTypeInvalid;
        }

        bool
        ClearBit (uint32_t bit);

        bool
        SetBit (uint32_t bit);

        bool
        operator == (const RegisterValue &rhs) const;

        bool
        operator != (const RegisterValue &rhs) const;

        void
        operator = (uint8_t uint)
        {
            m_type = eTypeUInt8;
            m_data.uint8 = uint;
        }

        void
        operator = (uint16_t uint)
        {
            m_type = eTypeUInt16;
            m_data.uint16 = uint;
        }

        void
        operator = (uint32_t uint)
        {
            m_type = eTypeUInt32;
            m_data.uint32 = uint;
        }

        void
        operator = (uint64_t uint)
        {
            m_type = eTypeUInt64;
            m_data.uint64 = uint;
        }

#if defined (ENABLE_128_BIT_SUPPORT)
        void
        operator = (__uint128_t uint)
        {
            m_type = eTypeUInt128;
            m_data.uint128 = uint;
        }
#endif        
        void
        operator = (float f)
        {
            m_type = eTypeFloat;
            m_data.ieee_float = f;
        }

        void
        operator = (double f)
        {
            m_type = eTypeDouble;
            m_data.ieee_double = f;
        }

        void
        operator = (long double f)
        {
            m_type = eTypeLongDouble;
            m_data.ieee_long_double = f;
        }

        void
        SetUInt8 (uint8_t uint)
        {
            m_type = eTypeUInt8;
            m_data.uint8 = uint;
        }

        void
        SetUInt16 (uint16_t uint)
        {
            m_type = eTypeUInt16;
            m_data.uint16 = uint;
        }

        void
        SetUInt32 (uint32_t uint, Type t = eTypeUInt32)
        {
            m_type = t;
            m_data.uint32 = uint;
        }

        void
        SetUInt64 (uint64_t uint, Type t = eTypeUInt64)
        {
            m_type = t;
            m_data.uint64 = uint;
        }

#if defined (ENABLE_128_BIT_SUPPORT)
        void
        SetUInt128 (__uint128_t uint)
        {
            m_type = eTypeUInt128;
            m_data.uint128 = uint;
        }
#endif
        bool
        SetUInt (uint64_t uint, uint32_t byte_size);
    
        void
        SetFloat (float f)
        {
            m_type = eTypeFloat;
            m_data.ieee_float = f;
        }

        void
        SetDouble (double f)
        {
            m_type = eTypeDouble;
            m_data.ieee_double = f;
        }

        void
        SetLongDouble (long double f)
        {
            m_type = eTypeLongDouble;
            m_data.ieee_long_double = f;
        }

        void
        SetBytes (const void *bytes, size_t length, lldb::ByteOrder byte_order);

        bool
        SignExtend (uint32_t sign_bitpos);

        Error
        SetValueFromCString (const RegisterInfo *reg_info, 
                             const char *value_str);

        Error
        SetValueFromData (const RegisterInfo *reg_info, 
                          DataExtractor &data, 
                          lldb::offset_t offset,
                          bool partial_data_ok);

        // The default value of 0 for reg_name_right_align_at means no alignment at all.
        bool
        Dump (Stream *s, 
              const RegisterInfo *reg_info, 
              bool prefix_with_name,
              bool prefix_with_alt_name,
              lldb::Format format,
              uint32_t reg_name_right_align_at = 0) const;

        void *
        GetBytes ();
        
        const void *
        GetBytes () const;

        lldb::ByteOrder
        GetByteOrder () const
        {
            if (m_type == eTypeBytes)
                return m_data.buffer.byte_order;
            return lldb::endian::InlHostByteOrder();
        }
        
        uint32_t
        GetByteSize () const;

        void
        Clear();

    protected:

        RegisterValue::Type m_type;
        union
        {
            uint8_t  uint8;
            uint16_t uint16;
            uint32_t uint32;
            uint64_t uint64;
#if defined (ENABLE_128_BIT_SUPPORT)
            __uint128_t uint128;
#endif
            float ieee_float;
            double ieee_double;
            long double ieee_long_double;
            struct 
            {
                uint8_t bytes[kMaxRegisterByteSize]; // This must be big enough to hold any register for any supported target.
                uint8_t length;
                lldb::ByteOrder byte_order;
            } buffer;
        } m_data;
    };

} // namespace lldb_private

#endif	// lldb_RegisterValue_h
