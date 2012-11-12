//===-- Scalar.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Scalar.h"

#include <math.h>

#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Host/Endian.h"

#include "Plugins/Process/Utility/InstructionUtils.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Promote to max type currently follows the ANSI C rule for type
// promotion in expressions.
//----------------------------------------------------------------------
static Scalar::Type
PromoteToMaxType
(
    const Scalar& lhs,                  // The const left hand side object
    const Scalar& rhs,                  // The const right hand side object
    Scalar& temp_value,             // A modifiable temp value than can be used to hold either the promoted lhs or rhs object
    const Scalar* &promoted_lhs_ptr,    // Pointer to the resulting possibly promoted value of lhs (at most one of lhs/rhs will get promoted)
    const Scalar* &promoted_rhs_ptr // Pointer to the resulting possibly promoted value of rhs (at most one of lhs/rhs will get promoted)
)
{
    Scalar result;
    // Initialize the promoted values for both the right and left hand side values
    // to be the objects themselves. If no promotion is needed (both right and left
    // have the same type), then the temp_value will not get used.
    promoted_lhs_ptr = &lhs;
    promoted_rhs_ptr = &rhs;
    // Extract the types of both the right and left hand side values
    Scalar::Type lhs_type = lhs.GetType();
    Scalar::Type rhs_type = rhs.GetType();

    if (lhs_type > rhs_type)
    {
        // Right hand side need to be promoted
        temp_value = rhs;                   // Copy right hand side into the temp value
        if (temp_value.Promote(lhs_type))   // Promote it
            promoted_rhs_ptr = &temp_value; // Update the pointer for the promoted right hand side
    }
    else if (lhs_type < rhs_type)
    {
        // Left hand side need to be promoted
        temp_value = lhs;                   // Copy left hand side value into the temp value
        if (temp_value.Promote(rhs_type))   // Promote it
            promoted_lhs_ptr = &temp_value; // Update the pointer for the promoted left hand side
    }

    // Make sure our type promotion worked as exptected
    if (promoted_lhs_ptr->GetType() == promoted_rhs_ptr->GetType())
        return promoted_lhs_ptr->GetType(); // Return the resulting max type

    // Return the void type (zero) if we fail to promote either of the values.
    return Scalar::e_void;
}


//----------------------------------------------------------------------
// Scalar constructor
//----------------------------------------------------------------------
Scalar::Scalar() :
    m_type(e_void),
    m_data()
{
}

//----------------------------------------------------------------------
// Scalar copy constructor
//----------------------------------------------------------------------
Scalar::Scalar(const Scalar& rhs) :
    m_type(rhs.m_type),
    m_data(rhs.m_data)  // TODO: verify that for C++ this will correctly copy the union??
{
}

//Scalar::Scalar(const RegisterValue& reg) :
//  m_type(e_void),
//  m_data()
//{
//  switch (reg.info.encoding)
//  {
//  case eEncodingUint:     // unsigned integer
//      switch (reg.info.byte_size)
//      {
//      case 1: m_type = e_uint; m_data.uint = reg.value.uint8; break;
//      case 2: m_type = e_uint; m_data.uint = reg.value.uint16; break;
//      case 4: m_type = e_uint; m_data.uint = reg.value.uint32; break;
//      case 8: m_type = e_ulonglong; m_data.ulonglong = reg.value.uint64; break;
//      break;
//      }
//      break;
//
//  case eEncodingSint:     // signed integer
//      switch (reg.info.byte_size)
//      {
//      case 1: m_type = e_sint; m_data.sint = reg.value.sint8; break;
//      case 2: m_type = e_sint; m_data.sint = reg.value.sint16; break;
//      case 4: m_type = e_sint; m_data.sint = reg.value.sint32; break;
//      case 8: m_type = e_slonglong; m_data.slonglong = reg.value.sint64; break;
//      break;
//      }
//      break;
//
//  case eEncodingIEEE754:  // float
//      switch (reg.info.byte_size)
//      {
//      case 4: m_type = e_float; m_data.flt = reg.value.float32; break;
//      case 8: m_type = e_double; m_data.dbl = reg.value.float64; break;
//      break;
//      }
//      break;
//    case eEncodingVector: // vector registers
//      break;
//  }
//}

bool
Scalar::GetData (DataExtractor &data, size_t limit_byte_size) const
{
    size_t byte_size = GetByteSize();
    if (byte_size > 0)
    {
        if (limit_byte_size < byte_size)
        {
            if (lldb::endian::InlHostByteOrder() == eByteOrderLittle)
            {
                // On little endian systems if we want fewer bytes from the
                // current type we just specify fewer bytes since the LSByte
                // is first...
                data.SetData((uint8_t*)&m_data, limit_byte_size, lldb::endian::InlHostByteOrder());
            }
            else if (lldb::endian::InlHostByteOrder() == eByteOrderBig)
            {
                // On big endian systems if we want fewer bytes from the
                // current type have to advance our initial byte pointer and
                // trim down the number of bytes since the MSByte is first
                data.SetData(((uint8_t*)&m_data) + byte_size - limit_byte_size, limit_byte_size, lldb::endian::InlHostByteOrder());
            }
        }
        else
        {
            // We want all of the data
            data.SetData((uint8_t*)&m_data, byte_size, lldb::endian::InlHostByteOrder());
        }
        return true;
    }
    data.Clear();
    return false;
}

size_t
Scalar::GetByteSize() const
{
    switch (m_type)
    {
    default:
    case e_void:
        break;
    case e_sint:        return sizeof(m_data.sint);
    case e_uint:        return sizeof(m_data.uint);
    case e_slong:       return sizeof(m_data.slong);
    case e_ulong:       return sizeof(m_data.ulong);
    case e_slonglong:   return sizeof(m_data.slonglong);
    case e_ulonglong:   return sizeof(m_data.ulonglong);
    case e_float:       return sizeof(m_data.flt);
    case e_double:      return sizeof(m_data.dbl);
    case e_long_double: return sizeof(m_data.ldbl);
    }
    return 0;
}

bool
Scalar::IsZero() const
{
    switch (m_type)
    {
    default:
    case e_void:
        break;
    case e_sint:        return m_data.sint == 0;
    case e_uint:        return m_data.uint == 0;
    case e_slong:       return m_data.slong == 0;
    case e_ulong:       return m_data.ulong == 0;
    case e_slonglong:   return m_data.slonglong == 0;
    case e_ulonglong:   return m_data.ulonglong == 0;
    case e_float:       return m_data.flt == 0.0f;
    case e_double:      return m_data.dbl == 0.0;
    case e_long_double: return m_data.ldbl == 0.0;
    }
    return false;
}

void
Scalar::GetValue (Stream *s, bool show_type) const
{
    if (show_type)
        s->Printf("(%s) ", GetTypeAsCString());

    switch (m_type)
    {
    case e_void:
    default:
        break;
    case e_sint:        s->Printf("%i", m_data.sint);               break;
    case e_uint:        s->Printf("0x%8.8x", m_data.uint);          break;
    case e_slong:       s->Printf("%li", m_data.slong);             break;
    case e_ulong:       s->Printf("0x%8.8lx", m_data.ulong);        break;
    case e_slonglong:   s->Printf("%lli", m_data.slonglong);        break;
    case e_ulonglong:   s->Printf("0x%16.16llx", m_data.ulonglong); break;
    case e_float:       s->Printf("%f", m_data.flt);                break;
    case e_double:      s->Printf("%g", m_data.dbl);                break;
    case e_long_double: s->Printf("%Lg", m_data.ldbl);              break;
    }
}

const char *
Scalar::GetTypeAsCString() const
{
    switch (m_type)
    {
    default:
        break;
    case e_void:        return "void";
    case e_sint:        return "int";
    case e_uint:        return "unsigned int";
    case e_slong:       return "long";
    case e_ulong:       return "unsigned long";
    case e_slonglong:   return "long long";
    case e_ulonglong:   return "unsigned long long";
    case e_float:       return "float";
    case e_double:      return "double";
    case e_long_double: return "long double";
    }
    return "<invalid Scalar type>";
}



//----------------------------------------------------------------------
// Scalar copy constructor
//----------------------------------------------------------------------
Scalar&
Scalar::operator=(const Scalar& rhs)
{
    if (this != &rhs)
    {
        m_type = rhs.m_type;
        ::memcpy (&m_data, &rhs.m_data, sizeof(m_data));
    }
    return *this;
}

Scalar&
Scalar::operator= (const int v)
{
    m_type = e_sint;
    m_data.sint = v;
    return *this;
}


Scalar&
Scalar::operator= (unsigned int v)
{
    m_type = e_uint;
    m_data.uint = v;
    return *this;
}

Scalar&
Scalar::operator= (long v)
{
    m_type = e_slong;
    m_data.slong = v;
    return *this;
}

Scalar&
Scalar::operator= (unsigned long v)
{
    m_type = e_ulong;
    m_data.ulong = v;
    return *this;
}

Scalar&
Scalar::operator= (long long v)
{
    m_type = e_slonglong;
    m_data.slonglong = v;
    return *this;
}

Scalar&
Scalar::operator= (unsigned long long v)
{
    m_type = e_ulonglong;
    m_data.ulonglong = v;
    return *this;
}

Scalar&
Scalar::operator= (float v)
{
    m_type = e_float;
    m_data.flt = v;
    return *this;
}

Scalar&
Scalar::operator= (double v)
{
    m_type = e_double;
    m_data.dbl = v;
    return *this;
}

Scalar&
Scalar::operator= (long double v)
{
    m_type = e_long_double;
    m_data.ldbl = v;
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Scalar::~Scalar()
{
}

bool
Scalar::Promote(Scalar::Type type)
{
    bool success = false;
    switch (m_type)
    {
    case e_void:
        break;

    case e_sint:
        switch (type)
        {
        default:
        case e_void:        break;
        case e_sint:        success = true; break;
        case e_uint:        m_data.uint         = m_data.sint;      success = true; break;
        case e_slong:       m_data.slong        = m_data.sint;      success = true; break;
        case e_ulong:       m_data.ulong        = m_data.sint;      success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.sint;      success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.sint;      success = true; break;
        case e_float:       m_data.flt          = m_data.sint;      success = true; break;
        case e_double:      m_data.dbl          = m_data.sint;      success = true; break;
        case e_long_double: m_data.ldbl         = m_data.sint;      success = true; break;
        }
        break;

    case e_uint:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:        break;
        case e_uint:        success = true; break;
        case e_slong:       m_data.slong        = m_data.uint;      success = true; break;
        case e_ulong:       m_data.ulong        = m_data.uint;      success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.uint;      success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.uint;      success = true; break;
        case e_float:       m_data.flt          = m_data.uint;      success = true; break;
        case e_double:      m_data.dbl          = m_data.uint;      success = true; break;
        case e_long_double: m_data.ldbl         = m_data.uint;      success = true; break;
        }
        break;

    case e_slong:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:
        case e_uint:        break;
        case e_slong:       success = true; break;
        case e_ulong:       m_data.ulong        = m_data.slong;     success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.slong;     success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.slong;     success = true; break;
        case e_float:       m_data.flt          = m_data.slong;     success = true; break;
        case e_double:      m_data.dbl          = m_data.slong;     success = true; break;
        case e_long_double: m_data.ldbl         = m_data.slong;     success = true; break;
        }
        break;

    case e_ulong:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:
        case e_uint:
        case e_slong:       break;
        case e_ulong:       success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.ulong;     success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.ulong;     success = true; break;
        case e_float:       m_data.flt          = m_data.ulong;     success = true; break;
        case e_double:      m_data.dbl          = m_data.ulong;     success = true; break;
        case e_long_double: m_data.ldbl         = m_data.ulong;     success = true; break;
        }
        break;

    case e_slonglong:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:
        case e_uint:
        case e_slong:
        case e_ulong:       break;
        case e_slonglong:   success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.slonglong;     success = true; break;
        case e_float:       m_data.flt          = m_data.slonglong;     success = true; break;
        case e_double:      m_data.dbl          = m_data.slonglong;     success = true; break;
        case e_long_double: m_data.ldbl         = m_data.slonglong;     success = true; break;
        }
        break;

    case e_ulonglong:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:
        case e_uint:
        case e_slong:
        case e_ulong:
        case e_slonglong:   break;
        case e_ulonglong:   success = true; break;
        case e_float:       m_data.flt          = m_data.ulonglong;     success = true; break;
        case e_double:      m_data.dbl          = m_data.ulonglong;     success = true; break;
        case e_long_double: m_data.ldbl         = m_data.ulonglong;     success = true; break;
        }
        break;

    case e_float:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:
        case e_uint:
        case e_slong:
        case e_ulong:
        case e_slonglong:
        case e_ulonglong:   break;
        case e_float:       success = true; break;
        case e_double:      m_data.dbl          = m_data.flt;           success = true; break;
        case e_long_double: m_data.ldbl         = m_data.ulonglong;     success = true; break;
        }
        break;

    case e_double:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:
        case e_uint:
        case e_slong:
        case e_ulong:
        case e_slonglong:
        case e_ulonglong:
        case e_float:       break;
        case e_double:      success = true; break;
        case e_long_double: m_data.ldbl         = m_data.dbl;       success = true; break;
        }
        break;

    case e_long_double:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:
        case e_uint:
        case e_slong:
        case e_ulong:
        case e_slonglong:
        case e_ulonglong:
        case e_float:
        case e_double:      break;
        case e_long_double: success = true; break;
        }
        break;
    }

    if (success)
        m_type = type;
    return success;
}

const char *
Scalar::GetValueTypeAsCString (Scalar::Type type)
{
    switch (type)
    {
    default:            break;
    case e_void:        return "void";
    case e_sint:        return "int";
    case e_uint:        return "unsigned int";
    case e_slong:       return "long";
    case e_ulong:       return "unsigned long";
    case e_slonglong:   return "long long";
    case e_ulonglong:   return "unsigned long long";
    case e_float:       return "float";
    case e_double:      return "double";
    case e_long_double: return "long double";
    }
    return "???";
}


Scalar::Type
Scalar::GetValueTypeForSignedIntegerWithByteSize (size_t byte_size)
{
    if (byte_size <= sizeof(int))
        return e_sint;
    if (byte_size <= sizeof(long))
        return e_slong;
    if (byte_size <= sizeof(long long))
        return e_slonglong;
    return e_void;
}

Scalar::Type
Scalar::GetValueTypeForUnsignedIntegerWithByteSize (size_t byte_size)
{
    if (byte_size <= sizeof(unsigned int))
        return e_uint;
    if (byte_size <= sizeof(unsigned long))
        return e_ulong;
    if (byte_size <= sizeof(unsigned long long))
        return e_ulonglong;
    return e_void;
}

Scalar::Type
Scalar::GetValueTypeForFloatWithByteSize (size_t byte_size)
{
    if (byte_size == sizeof(float))
        return e_float;
    if (byte_size == sizeof(double))
        return e_double;
    if (byte_size == sizeof(long double))
        return e_long_double;
    return e_void;
}

bool
Scalar::Cast(Scalar::Type type)
{
    bool success = false;
    switch (m_type)
    {
    case e_void:
        break;

    case e_sint:
        switch (type)
        {
        default:
        case e_void:        break;
        case e_sint:        success = true; break;
        case e_uint:        m_data.uint         = m_data.sint;      success = true; break;
        case e_slong:       m_data.slong        = m_data.sint;      success = true; break;
        case e_ulong:       m_data.ulong        = m_data.sint;      success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.sint;      success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.sint;      success = true; break;
        case e_float:       m_data.flt          = m_data.sint;      success = true; break;
        case e_double:      m_data.dbl          = m_data.sint;      success = true; break;
        case e_long_double: m_data.ldbl         = m_data.sint;      success = true; break;
        }
        break;

    case e_uint:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:        m_data.sint         = m_data.uint;      success = true; break;
        case e_uint:        success = true; break;
        case e_slong:       m_data.slong        = m_data.uint;      success = true; break;
        case e_ulong:       m_data.ulong        = m_data.uint;      success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.uint;      success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.uint;      success = true; break;
        case e_float:       m_data.flt          = m_data.uint;      success = true; break;
        case e_double:      m_data.dbl          = m_data.uint;      success = true; break;
        case e_long_double: m_data.ldbl         = m_data.uint;      success = true; break;
        }
        break;

    case e_slong:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:        m_data.sint         = m_data.slong;     success = true; break;
        case e_uint:        m_data.uint         = m_data.slong;     success = true; break;
        case e_slong:       success = true; break;
        case e_ulong:       m_data.ulong        = m_data.slong;     success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.slong;     success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.slong;     success = true; break;
        case e_float:       m_data.flt          = m_data.slong;     success = true; break;
        case e_double:      m_data.dbl          = m_data.slong;     success = true; break;
        case e_long_double: m_data.ldbl         = m_data.slong;     success = true; break;
        }
        break;

    case e_ulong:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:        m_data.sint         = m_data.ulong;     success = true; break;
        case e_uint:        m_data.uint         = m_data.ulong;     success = true; break;
        case e_slong:       m_data.slong        = m_data.ulong;     success = true; break;
        case e_ulong:       success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.ulong;     success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.ulong;     success = true; break;
        case e_float:       m_data.flt          = m_data.ulong;     success = true; break;
        case e_double:      m_data.dbl          = m_data.ulong;     success = true; break;
        case e_long_double: m_data.ldbl         = m_data.ulong;     success = true; break;
        }
        break;

    case e_slonglong:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:        m_data.sint         = m_data.slonglong;     success = true; break;
        case e_uint:        m_data.uint         = m_data.slonglong;     success = true; break;
        case e_slong:       m_data.slong        = m_data.slonglong;     success = true; break;
        case e_ulong:       m_data.ulong        = m_data.slonglong;     success = true; break;
        case e_slonglong:   success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.slonglong;     success = true; break;
        case e_float:       m_data.flt          = m_data.slonglong;     success = true; break;
        case e_double:      m_data.dbl          = m_data.slonglong;     success = true; break;
        case e_long_double: m_data.ldbl         = m_data.slonglong;     success = true; break;
        }
        break;

    case e_ulonglong:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:        m_data.sint         = m_data.ulonglong;     success = true; break;
        case e_uint:        m_data.uint         = m_data.ulonglong;     success = true; break;
        case e_slong:       m_data.slong        = m_data.ulonglong;     success = true; break;
        case e_ulong:       m_data.ulong        = m_data.ulonglong;     success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.ulonglong;     success = true; break;
        case e_ulonglong:   success = true; break;
        case e_float:       m_data.flt          = m_data.ulonglong;     success = true; break;
        case e_double:      m_data.dbl          = m_data.ulonglong;     success = true; break;
        case e_long_double: m_data.ldbl         = m_data.ulonglong;     success = true; break;
        }
        break;

    case e_float:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:        m_data.sint         = m_data.flt;       success = true; break;
        case e_uint:        m_data.uint         = m_data.flt;       success = true; break;
        case e_slong:       m_data.slong        = m_data.flt;       success = true; break;
        case e_ulong:       m_data.ulong        = m_data.flt;       success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.flt;       success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.flt;       success = true; break;
        case e_float:       success = true; break;
        case e_double:      m_data.dbl          = m_data.flt;       success = true; break;
        case e_long_double: m_data.ldbl         = m_data.flt;       success = true; break;
        }
        break;

    case e_double:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:        m_data.sint         = m_data.dbl;       success = true; break;
        case e_uint:        m_data.uint         = m_data.dbl;       success = true; break;
        case e_slong:       m_data.slong        = m_data.dbl;       success = true; break;
        case e_ulong:       m_data.ulong        = m_data.dbl;       success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.dbl;       success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.dbl;       success = true; break;
        case e_float:       m_data.flt          = m_data.dbl;       success = true; break;
        case e_double:      success = true; break;
        case e_long_double: m_data.ldbl         = m_data.dbl;       success = true; break;
        }
        break;

    case e_long_double:
        switch (type)
        {
        default:
        case e_void:
        case e_sint:        m_data.sint         = m_data.ldbl;      success = true; break;
        case e_uint:        m_data.uint         = m_data.ldbl;      success = true; break;
        case e_slong:       m_data.slong        = m_data.ldbl;      success = true; break;
        case e_ulong:       m_data.ulong        = m_data.ldbl;      success = true; break;
        case e_slonglong:   m_data.slonglong    = m_data.ldbl;      success = true; break;
        case e_ulonglong:   m_data.ulonglong    = m_data.ldbl;      success = true; break;
        case e_float:       m_data.flt          = m_data.ldbl;      success = true; break;
        case e_double:      m_data.dbl          = m_data.ldbl;      success = true; break;
        case e_long_double: success = true; break;
        }
        break;
    }

    if (success)
        m_type = type;
    return success;
}

int
Scalar::SInt(int fail_value) const
{
    switch (m_type)
    {
    default:
    case e_void:        break;
    case e_sint:        return m_data.sint;
    case e_uint:        return (int)m_data.uint;
    case e_slong:       return (int)m_data.slong;
    case e_ulong:       return (int)m_data.ulong;
    case e_slonglong:   return (int)m_data.slonglong;
    case e_ulonglong:   return (int)m_data.ulonglong;
    case e_float:       return (int)m_data.flt;
    case e_double:      return (int)m_data.dbl;
    case e_long_double: return (int)m_data.ldbl;
    }
    return fail_value;
}

unsigned int
Scalar::UInt(unsigned int fail_value) const
{
    switch (m_type)
    {
    default:
    case e_void:        break;
    case e_sint:        return (unsigned int)m_data.sint;
    case e_uint:        return (unsigned int)m_data.uint;
    case e_slong:       return (unsigned int)m_data.slong;
    case e_ulong:       return (unsigned int)m_data.ulong;
    case e_slonglong:   return (unsigned int)m_data.slonglong;
    case e_ulonglong:   return (unsigned int)m_data.ulonglong;
    case e_float:       return (unsigned int)m_data.flt;
    case e_double:      return (unsigned int)m_data.dbl;
    case e_long_double: return (unsigned int)m_data.ldbl;
    }
    return fail_value;
}


long
Scalar::SLong(long fail_value) const
{
    switch (m_type)
    {
    default:
    case e_void:        break;
    case e_sint:        return (long)m_data.sint;
    case e_uint:        return (long)m_data.uint;
    case e_slong:       return (long)m_data.slong;
    case e_ulong:       return (long)m_data.ulong;
    case e_slonglong:   return (long)m_data.slonglong;
    case e_ulonglong:   return (long)m_data.ulonglong;
    case e_float:       return (long)m_data.flt;
    case e_double:      return (long)m_data.dbl;
    case e_long_double: return (long)m_data.ldbl;
    }
    return fail_value;
}



unsigned long
Scalar::ULong(unsigned long fail_value) const
{
    switch (m_type)
    {
    default:
    case e_void:        break;
    case e_sint:        return (unsigned long)m_data.sint;
    case e_uint:        return (unsigned long)m_data.uint;
    case e_slong:       return (unsigned long)m_data.slong;
    case e_ulong:       return (unsigned long)m_data.ulong;
    case e_slonglong:   return (unsigned long)m_data.slonglong;
    case e_ulonglong:   return (unsigned long)m_data.ulonglong;
    case e_float:       return (unsigned long)m_data.flt;
    case e_double:      return (unsigned long)m_data.dbl;
    case e_long_double: return (unsigned long)m_data.ldbl;
    }
    return fail_value;
}

uint64_t
Scalar::GetRawBits64(uint64_t fail_value) const
{
    switch (m_type)
    {
    default:
    case e_void:
        break;

    case e_sint:
    case e_uint:
        return m_data.uint;

    case e_slong:
    case e_ulong:
        return m_data.ulong;

    case e_slonglong:
    case e_ulonglong:
        return m_data.ulonglong;

    case e_float:
        if (sizeof(m_data.flt) == sizeof(int))
            return m_data.uint;
        else if (sizeof(m_data.flt) == sizeof(unsigned long))
            return m_data.ulong;
        else if (sizeof(m_data.flt) == sizeof(unsigned long long))
            return m_data.ulonglong;
        break;

    case e_double:
        if (sizeof(m_data.dbl) == sizeof(int))
            return m_data.uint;
        else if (sizeof(m_data.dbl) == sizeof(unsigned long))
            return m_data.ulong;
        else if (sizeof(m_data.dbl) == sizeof(unsigned long long))
            return m_data.ulonglong;
        break;

    case e_long_double:
        if (sizeof(m_data.ldbl) == sizeof(int))
            return m_data.uint;
        else if (sizeof(m_data.ldbl) == sizeof(unsigned long))
            return m_data.ulong;
        else if (sizeof(m_data.ldbl) == sizeof(unsigned long long))
            return m_data.ulonglong;
        break;
    }
    return fail_value;
}



long long
Scalar::SLongLong(long long fail_value) const
{
    switch (m_type)
    {
    default:
    case e_void:        break;
    case e_sint:        return (long long)m_data.sint;
    case e_uint:        return (long long)m_data.uint;
    case e_slong:       return (long long)m_data.slong;
    case e_ulong:       return (long long)m_data.ulong;
    case e_slonglong:   return (long long)m_data.slonglong;
    case e_ulonglong:   return (long long)m_data.ulonglong;
    case e_float:       return (long long)m_data.flt;
    case e_double:      return (long long)m_data.dbl;
    case e_long_double: return (long long)m_data.ldbl;
    }
    return fail_value;
}


unsigned long long
Scalar::ULongLong(unsigned long long fail_value) const
{
    switch (m_type)
    {
    default:
    case e_void:        break;
    case e_sint:        return (unsigned long long)m_data.sint;
    case e_uint:        return (unsigned long long)m_data.uint;
    case e_slong:       return (unsigned long long)m_data.slong;
    case e_ulong:       return (unsigned long long)m_data.ulong;
    case e_slonglong:   return (unsigned long long)m_data.slonglong;
    case e_ulonglong:   return (unsigned long long)m_data.ulonglong;
    case e_float:       return (unsigned long long)m_data.flt;
    case e_double:      return (unsigned long long)m_data.dbl;
    case e_long_double: return (unsigned long long)m_data.ldbl;
    }
    return fail_value;
}


float
Scalar::Float(float fail_value) const
{
    switch (m_type)
    {
    default:
    case e_void:        break;
    case e_sint:        return (float)m_data.sint;
    case e_uint:        return (float)m_data.uint;
    case e_slong:       return (float)m_data.slong;
    case e_ulong:       return (float)m_data.ulong;
    case e_slonglong:   return (float)m_data.slonglong;
    case e_ulonglong:   return (float)m_data.ulonglong;
    case e_float:       return (float)m_data.flt;
    case e_double:      return (float)m_data.dbl;
    case e_long_double: return (float)m_data.ldbl;
    }
    return fail_value;
}


double
Scalar::Double(double fail_value) const
{
    switch (m_type)
    {
    default:
    case e_void:        break;
    case e_sint:        return (double)m_data.sint;
    case e_uint:        return (double)m_data.uint;
    case e_slong:       return (double)m_data.slong;
    case e_ulong:       return (double)m_data.ulong;
    case e_slonglong:   return (double)m_data.slonglong;
    case e_ulonglong:   return (double)m_data.ulonglong;
    case e_float:       return (double)m_data.flt;
    case e_double:      return (double)m_data.dbl;
    case e_long_double: return (double)m_data.ldbl;
    }
    return fail_value;
}


long double
Scalar::LongDouble(long double fail_value) const
{
    switch (m_type)
    {
    default:
    case e_void:        break;
    case e_sint:        return (long double)m_data.sint;
    case e_uint:        return (long double)m_data.uint;
    case e_slong:       return (long double)m_data.slong;
    case e_ulong:       return (long double)m_data.ulong;
    case e_slonglong:   return (long double)m_data.slonglong;
    case e_ulonglong:   return (long double)m_data.ulonglong;
    case e_float:       return (long double)m_data.flt;
    case e_double:      return (long double)m_data.dbl;
    case e_long_double: return (long double)m_data.ldbl;
    }
    return fail_value;
}


Scalar&
Scalar::operator+= (const Scalar& rhs)
{
    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    if ((m_type = PromoteToMaxType(*this, rhs, temp_value, a, b)) != Scalar::e_void)
    {
        switch (m_type)
        {
        default:
        case e_void:        break;
        case e_sint:        m_data.sint         = a->m_data.sint        + b->m_data.sint;       break;
        case e_uint:        m_data.uint         = a->m_data.uint        + b->m_data.uint;       break;
        case e_slong:       m_data.slong        = a->m_data.slong       + b->m_data.slong;      break;
        case e_ulong:       m_data.ulong        = a->m_data.ulong       + b->m_data.ulong;      break;
        case e_slonglong:   m_data.slonglong    = a->m_data.slonglong   + b->m_data.slonglong;  break;
        case e_ulonglong:   m_data.ulonglong    = a->m_data.ulonglong   + b->m_data.ulonglong;  break;
        case e_float:       m_data.flt          = a->m_data.flt         + b->m_data.flt;        break;
        case e_double:      m_data.dbl          = a->m_data.dbl         + b->m_data.dbl;        break;
        case e_long_double: m_data.ldbl         = a->m_data.ldbl        + b->m_data.ldbl;       break;
        }
    }
    return *this;
}

Scalar&
Scalar::operator<<= (const Scalar& rhs)
{
    switch (m_type)
    {
    default:
    case e_void:
    case e_float:
    case e_double:
    case e_long_double:
        m_type = e_void;
        break;

    case e_sint:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.sint <<= rhs.m_data.sint;        break;
        case e_uint:            m_data.sint <<= rhs.m_data.uint;        break;
        case e_slong:           m_data.sint <<= rhs.m_data.slong;       break;
        case e_ulong:           m_data.sint <<= rhs.m_data.ulong;       break;
        case e_slonglong:       m_data.sint <<= rhs.m_data.slonglong;   break;
        case e_ulonglong:       m_data.sint <<= rhs.m_data.ulonglong;   break;
        }
        break;

    case e_uint:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.uint <<= rhs.m_data.sint;        break;
        case e_uint:            m_data.uint <<= rhs.m_data.uint;        break;
        case e_slong:           m_data.uint <<= rhs.m_data.slong;       break;
        case e_ulong:           m_data.uint <<= rhs.m_data.ulong;       break;
        case e_slonglong:       m_data.uint <<= rhs.m_data.slonglong;   break;
        case e_ulonglong:       m_data.uint <<= rhs.m_data.ulonglong;   break;
        }
        break;

    case e_slong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.slong <<= rhs.m_data.sint;       break;
        case e_uint:            m_data.slong <<= rhs.m_data.uint;       break;
        case e_slong:           m_data.slong <<= rhs.m_data.slong;      break;
        case e_ulong:           m_data.slong <<= rhs.m_data.ulong;      break;
        case e_slonglong:       m_data.slong <<= rhs.m_data.slonglong;  break;
        case e_ulonglong:       m_data.slong <<= rhs.m_data.ulonglong;  break;
        }
        break;

    case e_ulong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.ulong <<= rhs.m_data.sint;       break;
        case e_uint:            m_data.ulong <<= rhs.m_data.uint;       break;
        case e_slong:           m_data.ulong <<= rhs.m_data.slong;      break;
        case e_ulong:           m_data.ulong <<= rhs.m_data.ulong;      break;
        case e_slonglong:       m_data.ulong <<= rhs.m_data.slonglong;  break;
        case e_ulonglong:       m_data.ulong <<= rhs.m_data.ulonglong;  break;
        }
        break;
    case e_slonglong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.slonglong <<= rhs.m_data.sint;       break;
        case e_uint:            m_data.slonglong <<= rhs.m_data.uint;       break;
        case e_slong:           m_data.slonglong <<= rhs.m_data.slong;      break;
        case e_ulong:           m_data.slonglong <<= rhs.m_data.ulong;      break;
        case e_slonglong:       m_data.slonglong <<= rhs.m_data.slonglong;  break;
        case e_ulonglong:       m_data.slonglong <<= rhs.m_data.ulonglong;  break;
        }
        break;

    case e_ulonglong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.ulonglong <<= rhs.m_data.sint;       break;
        case e_uint:            m_data.ulonglong <<= rhs.m_data.uint;       break;
        case e_slong:           m_data.ulonglong <<= rhs.m_data.slong;      break;
        case e_ulong:           m_data.ulonglong <<= rhs.m_data.ulong;      break;
        case e_slonglong:       m_data.ulonglong <<= rhs.m_data.slonglong;  break;
        case e_ulonglong:       m_data.ulonglong <<= rhs.m_data.ulonglong;  break;
        }
        break;
    }
    return *this;
}

bool
Scalar::ShiftRightLogical(const Scalar& rhs)
{
    switch (m_type)
    {
    default:
    case e_void:
    case e_float:
    case e_double:
    case e_long_double:
        m_type = e_void;
        break;

    case e_sint:
    case e_uint:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.uint >>= rhs.m_data.sint;        break;
        case e_uint:            m_data.uint >>= rhs.m_data.uint;        break;
        case e_slong:           m_data.uint >>= rhs.m_data.slong;       break;
        case e_ulong:           m_data.uint >>= rhs.m_data.ulong;       break;
        case e_slonglong:       m_data.uint >>= rhs.m_data.slonglong;   break;
        case e_ulonglong:       m_data.uint >>= rhs.m_data.ulonglong;   break;
        }
        break;

    case e_slong:
    case e_ulong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.ulong >>= rhs.m_data.sint;       break;
        case e_uint:            m_data.ulong >>= rhs.m_data.uint;       break;
        case e_slong:           m_data.ulong >>= rhs.m_data.slong;      break;
        case e_ulong:           m_data.ulong >>= rhs.m_data.ulong;      break;
        case e_slonglong:       m_data.ulong >>= rhs.m_data.slonglong;  break;
        case e_ulonglong:       m_data.ulong >>= rhs.m_data.ulonglong;  break;
        }
        break;

    case e_slonglong:
    case e_ulonglong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.ulonglong >>= rhs.m_data.sint;       break;
        case e_uint:            m_data.ulonglong >>= rhs.m_data.uint;       break;
        case e_slong:           m_data.ulonglong >>= rhs.m_data.slong;      break;
        case e_ulong:           m_data.ulonglong >>= rhs.m_data.ulong;      break;
        case e_slonglong:       m_data.ulonglong >>= rhs.m_data.slonglong;  break;
        case e_ulonglong:       m_data.ulonglong >>= rhs.m_data.ulonglong;  break;
        }
        break;
    }
    return m_type != e_void;
}


Scalar&
Scalar::operator>>= (const Scalar& rhs)
{
    switch (m_type)
    {
    default:
    case e_void:
    case e_float:
    case e_double:
    case e_long_double:
        m_type = e_void;
        break;

    case e_sint:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.sint >>= rhs.m_data.sint;        break;
        case e_uint:            m_data.sint >>= rhs.m_data.uint;        break;
        case e_slong:           m_data.sint >>= rhs.m_data.slong;       break;
        case e_ulong:           m_data.sint >>= rhs.m_data.ulong;       break;
        case e_slonglong:       m_data.sint >>= rhs.m_data.slonglong;   break;
        case e_ulonglong:       m_data.sint >>= rhs.m_data.ulonglong;   break;
        }
        break;

    case e_uint:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.uint >>= rhs.m_data.sint;        break;
        case e_uint:            m_data.uint >>= rhs.m_data.uint;        break;
        case e_slong:           m_data.uint >>= rhs.m_data.slong;       break;
        case e_ulong:           m_data.uint >>= rhs.m_data.ulong;       break;
        case e_slonglong:       m_data.uint >>= rhs.m_data.slonglong;   break;
        case e_ulonglong:       m_data.uint >>= rhs.m_data.ulonglong;   break;
        }
        break;

    case e_slong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.slong >>= rhs.m_data.sint;       break;
        case e_uint:            m_data.slong >>= rhs.m_data.uint;       break;
        case e_slong:           m_data.slong >>= rhs.m_data.slong;      break;
        case e_ulong:           m_data.slong >>= rhs.m_data.ulong;      break;
        case e_slonglong:       m_data.slong >>= rhs.m_data.slonglong;  break;
        case e_ulonglong:       m_data.slong >>= rhs.m_data.ulonglong;  break;
        }
        break;

    case e_ulong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.ulong >>= rhs.m_data.sint;       break;
        case e_uint:            m_data.ulong >>= rhs.m_data.uint;       break;
        case e_slong:           m_data.ulong >>= rhs.m_data.slong;      break;
        case e_ulong:           m_data.ulong >>= rhs.m_data.ulong;      break;
        case e_slonglong:       m_data.ulong >>= rhs.m_data.slonglong;  break;
        case e_ulonglong:       m_data.ulong >>= rhs.m_data.ulonglong;  break;
        }
        break;
    case e_slonglong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.slonglong >>= rhs.m_data.sint;       break;
        case e_uint:            m_data.slonglong >>= rhs.m_data.uint;       break;
        case e_slong:           m_data.slonglong >>= rhs.m_data.slong;      break;
        case e_ulong:           m_data.slonglong >>= rhs.m_data.ulong;      break;
        case e_slonglong:       m_data.slonglong >>= rhs.m_data.slonglong;  break;
        case e_ulonglong:       m_data.slonglong >>= rhs.m_data.ulonglong;  break;
        }
        break;

    case e_ulonglong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.ulonglong >>= rhs.m_data.sint;       break;
        case e_uint:            m_data.ulonglong >>= rhs.m_data.uint;       break;
        case e_slong:           m_data.ulonglong >>= rhs.m_data.slong;      break;
        case e_ulong:           m_data.ulonglong >>= rhs.m_data.ulong;      break;
        case e_slonglong:       m_data.ulonglong >>= rhs.m_data.slonglong;  break;
        case e_ulonglong:       m_data.ulonglong >>= rhs.m_data.ulonglong;  break;
        }
        break;
    }
    return *this;
}


Scalar&
Scalar::operator&= (const Scalar& rhs)
{
    switch (m_type)
    {
    default:
    case e_void:
    case e_float:
    case e_double:
    case e_long_double:
        m_type = e_void;
        break;

    case e_sint:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.sint &= rhs.m_data.sint;         break;
        case e_uint:            m_data.sint &= rhs.m_data.uint;         break;
        case e_slong:           m_data.sint &= rhs.m_data.slong;        break;
        case e_ulong:           m_data.sint &= rhs.m_data.ulong;        break;
        case e_slonglong:       m_data.sint &= rhs.m_data.slonglong;    break;
        case e_ulonglong:       m_data.sint &= rhs.m_data.ulonglong;    break;
        }
        break;

    case e_uint:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.uint &= rhs.m_data.sint;         break;
        case e_uint:            m_data.uint &= rhs.m_data.uint;         break;
        case e_slong:           m_data.uint &= rhs.m_data.slong;        break;
        case e_ulong:           m_data.uint &= rhs.m_data.ulong;        break;
        case e_slonglong:       m_data.uint &= rhs.m_data.slonglong;    break;
        case e_ulonglong:       m_data.uint &= rhs.m_data.ulonglong;    break;
        }
        break;

    case e_slong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.slong &= rhs.m_data.sint;        break;
        case e_uint:            m_data.slong &= rhs.m_data.uint;        break;
        case e_slong:           m_data.slong &= rhs.m_data.slong;       break;
        case e_ulong:           m_data.slong &= rhs.m_data.ulong;       break;
        case e_slonglong:       m_data.slong &= rhs.m_data.slonglong;   break;
        case e_ulonglong:       m_data.slong &= rhs.m_data.ulonglong;   break;
        }
        break;

    case e_ulong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.ulong &= rhs.m_data.sint;        break;
        case e_uint:            m_data.ulong &= rhs.m_data.uint;        break;
        case e_slong:           m_data.ulong &= rhs.m_data.slong;       break;
        case e_ulong:           m_data.ulong &= rhs.m_data.ulong;       break;
        case e_slonglong:       m_data.ulong &= rhs.m_data.slonglong;   break;
        case e_ulonglong:       m_data.ulong &= rhs.m_data.ulonglong;   break;
        }
        break;
    case e_slonglong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.slonglong &= rhs.m_data.sint;        break;
        case e_uint:            m_data.slonglong &= rhs.m_data.uint;        break;
        case e_slong:           m_data.slonglong &= rhs.m_data.slong;       break;
        case e_ulong:           m_data.slonglong &= rhs.m_data.ulong;       break;
        case e_slonglong:       m_data.slonglong &= rhs.m_data.slonglong;   break;
        case e_ulonglong:       m_data.slonglong &= rhs.m_data.ulonglong;   break;
        }
        break;

    case e_ulonglong:
        switch (rhs.m_type)
        {
        default:
        case e_void:
        case e_float:
        case e_double:
        case e_long_double:
            m_type = e_void;
            break;
        case e_sint:            m_data.ulonglong &= rhs.m_data.sint;        break;
        case e_uint:            m_data.ulonglong &= rhs.m_data.uint;        break;
        case e_slong:           m_data.ulonglong &= rhs.m_data.slong;       break;
        case e_ulong:           m_data.ulonglong &= rhs.m_data.ulong;       break;
        case e_slonglong:       m_data.ulonglong &= rhs.m_data.slonglong;   break;
        case e_ulonglong:       m_data.ulonglong &= rhs.m_data.ulonglong;   break;
        }
        break;
    }
    return *this;
}



bool
Scalar::AbsoluteValue()
{
    switch (m_type)
    {
    default:
    case e_void:
        break;

    case e_sint:
        if (m_data.sint < 0)
            m_data.sint = -m_data.sint;
        return true;

    case e_slong:
        if (m_data.slong < 0)
            m_data.slong = -m_data.slong;
        return true;

    case e_slonglong:
        if (m_data.slonglong < 0)
            m_data.slonglong = -m_data.slonglong;
        return true;

    case e_uint:
    case e_ulong:
    case e_ulonglong:   return true;
    case e_float:       m_data.flt = fabsf(m_data.flt);     return true;
    case e_double:      m_data.dbl = fabs(m_data.dbl);      return true;
    case e_long_double: m_data.ldbl = fabsl(m_data.ldbl);   return true;
    }
    return false;
}


bool
Scalar::UnaryNegate()
{
    switch (m_type)
    {
    default:
    case e_void:        break;
    case e_sint:        m_data.sint = -m_data.sint;             return true;
    case e_uint:        m_data.uint = -m_data.uint;             return true;
    case e_slong:       m_data.slong = -m_data.slong;           return true;
    case e_ulong:       m_data.ulong = -m_data.ulong;           return true;
    case e_slonglong:   m_data.slonglong = -m_data.slonglong;   return true;
    case e_ulonglong:   m_data.ulonglong = -m_data.ulonglong;   return true;
    case e_float:       m_data.flt = -m_data.flt;               return true;
    case e_double:      m_data.dbl = -m_data.dbl;               return true;
    case e_long_double: m_data.ldbl = -m_data.ldbl;             return true;
    }
    return false;
}

bool
Scalar::OnesComplement()
{
    switch (m_type)
    {
    case e_sint:        m_data.sint = ~m_data.sint; return true;
    case e_uint:        m_data.uint = ~m_data.uint; return true;
    case e_slong:       m_data.slong = ~m_data.slong; return true;
    case e_ulong:       m_data.ulong = ~m_data.ulong; return true;
    case e_slonglong:   m_data.slonglong = ~m_data.slonglong; return true;
    case e_ulonglong:   m_data.ulonglong = ~m_data.ulonglong; return true;

    default:
    case e_void:
    case e_float:
    case e_double:
    case e_long_double:
        break;
    }
    return false;
}


const Scalar
lldb_private::operator+ (const Scalar& lhs, const Scalar& rhs)
{
    Scalar result;
    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    if ((result.m_type = PromoteToMaxType(lhs, rhs, temp_value, a, b)) != Scalar::e_void)
    {
        switch (result.m_type)
        {
        default:
        case Scalar::e_void:            break;
        case Scalar::e_sint:            result.m_data.sint      = a->m_data.sint        + b->m_data.sint;       break;
        case Scalar::e_uint:            result.m_data.uint      = a->m_data.uint        + b->m_data.uint;       break;
        case Scalar::e_slong:           result.m_data.slong     = a->m_data.slong       + b->m_data.slong;      break;
        case Scalar::e_ulong:           result.m_data.ulong     = a->m_data.ulong       + b->m_data.ulong;      break;
        case Scalar::e_slonglong:       result.m_data.slonglong = a->m_data.slonglong   + b->m_data.slonglong;  break;
        case Scalar::e_ulonglong:       result.m_data.ulonglong = a->m_data.ulonglong   + b->m_data.ulonglong;  break;
        case Scalar::e_float:           result.m_data.flt       = a->m_data.flt         + b->m_data.flt;        break;
        case Scalar::e_double:      result.m_data.dbl       = a->m_data.dbl         + b->m_data.dbl;        break;
        case Scalar::e_long_double: result.m_data.ldbl      = a->m_data.ldbl        + b->m_data.ldbl;       break;
        }
    }
    return result;
}


const Scalar
lldb_private::operator- (const Scalar& lhs, const Scalar& rhs)
{
    Scalar result;
    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    if ((result.m_type = PromoteToMaxType(lhs, rhs, temp_value, a, b)) != Scalar::e_void)
    {
        switch (result.m_type)
        {
        default:
        case Scalar::e_void:            break;
        case Scalar::e_sint:            result.m_data.sint      = a->m_data.sint        - b->m_data.sint;       break;
        case Scalar::e_uint:            result.m_data.uint      = a->m_data.uint        - b->m_data.uint;       break;
        case Scalar::e_slong:           result.m_data.slong     = a->m_data.slong       - b->m_data.slong;      break;
        case Scalar::e_ulong:           result.m_data.ulong     = a->m_data.ulong       - b->m_data.ulong;      break;
        case Scalar::e_slonglong:       result.m_data.slonglong = a->m_data.slonglong   - b->m_data.slonglong;  break;
        case Scalar::e_ulonglong:       result.m_data.ulonglong = a->m_data.ulonglong   - b->m_data.ulonglong;  break;
        case Scalar::e_float:           result.m_data.flt       = a->m_data.flt         - b->m_data.flt;        break;
        case Scalar::e_double:      result.m_data.dbl       = a->m_data.dbl         - b->m_data.dbl;        break;
        case Scalar::e_long_double: result.m_data.ldbl      = a->m_data.ldbl        - b->m_data.ldbl;       break;
        }
    }
    return result;
}

const Scalar
lldb_private::operator/ (const Scalar& lhs, const Scalar& rhs)
{
    Scalar result;
    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    if ((result.m_type = PromoteToMaxType(lhs, rhs, temp_value, a, b)) != Scalar::e_void)
    {
        switch (result.m_type)
        {
        default:
        case Scalar::e_void:            break;

        case Scalar::e_sint:            if (b->m_data.sint != 0)        { result.m_data.sint = a->m_data.sint/ b->m_data.sint; return result; } break;
        case Scalar::e_uint:            if (b->m_data.uint != 0)        { result.m_data.uint = a->m_data.uint / b->m_data.uint; return result; } break;
        case Scalar::e_slong:           if (b->m_data.slong != 0)       { result.m_data.slong = a->m_data.slong / b->m_data.slong; return result; } break;
        case Scalar::e_ulong:           if (b->m_data.ulong != 0)       { result.m_data.ulong = a->m_data.ulong / b->m_data.ulong; return result; } break;
        case Scalar::e_slonglong:       if (b->m_data.slonglong != 0)   { result.m_data.slonglong = a->m_data.slonglong / b->m_data.slonglong; return result; } break;
        case Scalar::e_ulonglong:       if (b->m_data.ulonglong != 0)   { result.m_data.ulonglong = a->m_data.ulonglong / b->m_data.ulonglong; return result; } break;
        case Scalar::e_float:           if (b->m_data.flt != 0.0f)      { result.m_data.flt = a->m_data.flt / b->m_data.flt; return result; } break;
        case Scalar::e_double:      if (b->m_data.dbl != 0.0)       { result.m_data.dbl = a->m_data.dbl / b->m_data.dbl; return result; } break;
        case Scalar::e_long_double: if (b->m_data.ldbl != 0.0)      { result.m_data.ldbl = a->m_data.ldbl / b->m_data.ldbl; return result; } break;
        }
    }
    // For division only, the only way it should make it here is if a promotion failed,
    // or if we are trying to do a divide by zero.
    result.m_type = Scalar::e_void;
    return result;
}

const Scalar
lldb_private::operator* (const Scalar& lhs, const Scalar& rhs)
{
    Scalar result;
    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    if ((result.m_type = PromoteToMaxType(lhs, rhs, temp_value, a, b)) != Scalar::e_void)
    {
        switch (result.m_type)
        {
        default:
        case Scalar::e_void:            break;
        case Scalar::e_sint:            result.m_data.sint      = a->m_data.sint        * b->m_data.sint;       break;
        case Scalar::e_uint:            result.m_data.uint      = a->m_data.uint        * b->m_data.uint;       break;
        case Scalar::e_slong:           result.m_data.slong     = a->m_data.slong       * b->m_data.slong;      break;
        case Scalar::e_ulong:           result.m_data.ulong     = a->m_data.ulong       * b->m_data.ulong;      break;
        case Scalar::e_slonglong:       result.m_data.slonglong = a->m_data.slonglong   * b->m_data.slonglong;  break;
        case Scalar::e_ulonglong:       result.m_data.ulonglong = a->m_data.ulonglong   * b->m_data.ulonglong;  break;
        case Scalar::e_float:           result.m_data.flt       = a->m_data.flt         * b->m_data.flt;        break;
        case Scalar::e_double:      result.m_data.dbl       = a->m_data.dbl         * b->m_data.dbl;        break;
        case Scalar::e_long_double: result.m_data.ldbl      = a->m_data.ldbl        * b->m_data.ldbl;       break;
        }
    }
    return result;
}

const Scalar
lldb_private::operator& (const Scalar& lhs, const Scalar& rhs)
{
    Scalar result;
    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    if ((result.m_type = PromoteToMaxType(lhs, rhs, temp_value, a, b)) != Scalar::e_void)
    {
        switch (result.m_type)
        {
        case Scalar::e_sint:        result.m_data.sint      = a->m_data.sint        & b->m_data.sint;       break;
        case Scalar::e_uint:        result.m_data.uint      = a->m_data.uint        & b->m_data.uint;       break;
        case Scalar::e_slong:       result.m_data.slong     = a->m_data.slong       & b->m_data.slong;      break;
        case Scalar::e_ulong:       result.m_data.ulong     = a->m_data.ulong       & b->m_data.ulong;      break;
        case Scalar::e_slonglong:   result.m_data.slonglong = a->m_data.slonglong   & b->m_data.slonglong;  break;
        case Scalar::e_ulonglong:   result.m_data.ulonglong = a->m_data.ulonglong   & b->m_data.ulonglong;  break;

        default:
        case Scalar::e_void:
        case Scalar::e_float:
        case Scalar::e_double:
        case Scalar::e_long_double:
            // No bitwise AND on floats, doubles of long doubles
            result.m_type = Scalar::e_void;
            break;
        }
    }
    return result;
}

const Scalar
lldb_private::operator| (const Scalar& lhs, const Scalar& rhs)
{
    Scalar result;
    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    if ((result.m_type = PromoteToMaxType(lhs, rhs, temp_value, a, b)) != Scalar::e_void)
    {
        switch (result.m_type)
        {
        case Scalar::e_sint:        result.m_data.sint      = a->m_data.sint        | b->m_data.sint;       break;
        case Scalar::e_uint:        result.m_data.uint      = a->m_data.uint        | b->m_data.uint;       break;
        case Scalar::e_slong:       result.m_data.slong     = a->m_data.slong       | b->m_data.slong;      break;
        case Scalar::e_ulong:       result.m_data.ulong     = a->m_data.ulong       | b->m_data.ulong;      break;
        case Scalar::e_slonglong:   result.m_data.slonglong = a->m_data.slonglong   | b->m_data.slonglong;  break;
        case Scalar::e_ulonglong:   result.m_data.ulonglong = a->m_data.ulonglong   | b->m_data.ulonglong;  break;

        default:
        case Scalar::e_void:
        case Scalar::e_float:
        case Scalar::e_double:
        case Scalar::e_long_double:
            // No bitwise AND on floats, doubles of long doubles
            result.m_type = Scalar::e_void;
            break;
        }
    }
    return result;
}

const Scalar
lldb_private::operator% (const Scalar& lhs, const Scalar& rhs)
{
    Scalar result;
    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    if ((result.m_type = PromoteToMaxType(lhs, rhs, temp_value, a, b)) != Scalar::e_void)
    {
        switch (result.m_type)
        {
        case Scalar::e_sint:        result.m_data.sint      = a->m_data.sint        % b->m_data.sint;       break;
        case Scalar::e_uint:        result.m_data.uint      = a->m_data.uint        % b->m_data.uint;       break;
        case Scalar::e_slong:       result.m_data.slong     = a->m_data.slong       % b->m_data.slong;      break;
        case Scalar::e_ulong:       result.m_data.ulong     = a->m_data.ulong       % b->m_data.ulong;      break;
        case Scalar::e_slonglong:   result.m_data.slonglong = a->m_data.slonglong   % b->m_data.slonglong;  break;
        case Scalar::e_ulonglong:   result.m_data.ulonglong = a->m_data.ulonglong   % b->m_data.ulonglong;  break;

        default:
        case Scalar::e_void:
        case Scalar::e_float:
        case Scalar::e_double:
        case Scalar::e_long_double:
            // No bitwise AND on floats, doubles of long doubles
            result.m_type = Scalar::e_void;
            break;
        }
    }
    return result;
}

const Scalar
lldb_private::operator^ (const Scalar& lhs, const Scalar& rhs)
{
    Scalar result;
    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    if ((result.m_type = PromoteToMaxType(lhs, rhs, temp_value, a, b)) != Scalar::e_void)
    {
        switch (result.m_type)
        {
        case Scalar::e_sint:        result.m_data.sint      = a->m_data.sint        ^ b->m_data.sint;       break;
        case Scalar::e_uint:        result.m_data.uint      = a->m_data.uint        ^ b->m_data.uint;       break;
        case Scalar::e_slong:       result.m_data.slong     = a->m_data.slong       ^ b->m_data.slong;      break;
        case Scalar::e_ulong:       result.m_data.ulong     = a->m_data.ulong       ^ b->m_data.ulong;      break;
        case Scalar::e_slonglong:   result.m_data.slonglong = a->m_data.slonglong   ^ b->m_data.slonglong;  break;
        case Scalar::e_ulonglong:   result.m_data.ulonglong = a->m_data.ulonglong   ^ b->m_data.ulonglong;  break;

        default:
        case Scalar::e_void:
        case Scalar::e_float:
        case Scalar::e_double:
        case Scalar::e_long_double:
            // No bitwise AND on floats, doubles of long doubles
            result.m_type = Scalar::e_void;
            break;
        }
    }
    return result;
}

// Return the raw unsigned integer without any casting or conversion
unsigned int
Scalar::RawUInt () const
{
    return m_data.uint;
}

// Return the raw unsigned long without any casting or conversion
unsigned long
Scalar::RawULong () const
{
    return m_data.ulong;
}

// Return the raw unsigned long long without any casting or conversion
unsigned long long
Scalar::RawULongLong () const
{
    return m_data.ulonglong;
}


Error
Scalar::SetValueFromCString (const char *value_str, Encoding encoding, uint32_t byte_size)
{
    Error error;
    if (value_str == NULL || value_str[0] == '\0')
    {
        error.SetErrorString ("Invalid c-string value string.");
        return error;
    }
    bool success = false;
    switch (encoding)
    {
    default:
    case eEncodingInvalid:
        error.SetErrorString ("Invalid encoding.");
        break;

    case eEncodingUint:
        if (byte_size <= sizeof (unsigned long long))
        {
            uint64_t uval64 = Args::StringToUInt64(value_str, UINT64_MAX, 0, &success);
            if (!success)
                error.SetErrorStringWithFormat ("'%s' is not a valid unsigned integer string value", value_str);
            else if (!UIntValueIsValidForSize (uval64, byte_size))
                error.SetErrorStringWithFormat ("value 0x%llx is too large to fit in a %u byte unsigned integer value", uval64, byte_size);
            else
            {
                m_type = Scalar::GetValueTypeForUnsignedIntegerWithByteSize (byte_size);
                switch (m_type)
                {
                case e_uint:        m_data.uint = uval64;       break;
                case e_ulong:       m_data.ulong = uval64;      break;
                case e_ulonglong:   m_data.ulonglong = uval64;  break;
                default:
                    error.SetErrorStringWithFormat ("unsupported unsigned integer byte size: %u", byte_size);
                    break;
                }
            }
        }
        else
        {
            error.SetErrorStringWithFormat ("unsupported unsigned integer byte size: %u", byte_size);
            return error;
        }
        break;

    case eEncodingSint:
        if (byte_size <= sizeof (long long))
        {
            uint64_t sval64 = Args::StringToSInt64(value_str, INT64_MAX, 0, &success);
            if (!success)
                error.SetErrorStringWithFormat ("'%s' is not a valid signed integer string value", value_str);
            else if (!SIntValueIsValidForSize (sval64, byte_size))
                error.SetErrorStringWithFormat ("value 0x%llx is too large to fit in a %u byte signed integer value", sval64, byte_size);
            else
            {
                m_type = Scalar::GetValueTypeForSignedIntegerWithByteSize (byte_size);
                switch (m_type)
                {
                case e_sint:        m_data.sint = sval64;       break;
                case e_slong:       m_data.slong = sval64;      break;
                case e_slonglong:   m_data.slonglong = sval64;  break;
                default:
                    error.SetErrorStringWithFormat ("unsupported signed integer byte size: %u", byte_size);
                    break;
                }
            }
        }
        else
        {
            error.SetErrorStringWithFormat ("unsupported signed integer byte size: %u", byte_size);
            return error;
        }
        break;

    case eEncodingIEEE754:
        if (byte_size == sizeof (float))
        {
            if (::sscanf (value_str, "%f", &m_data.flt) == 1)
                m_type = e_float;
            else
                error.SetErrorStringWithFormat ("'%s' is not a valid float string value", value_str);
        }
        else if (byte_size == sizeof (double))
        {
            if (::sscanf (value_str, "%lf", &m_data.dbl) == 1)
                m_type = e_double;
            else
                error.SetErrorStringWithFormat ("'%s' is not a valid float string value", value_str);
        }
        else if (byte_size == sizeof (long double))
        {
            if (::sscanf (value_str, "%Lf", &m_data.ldbl) == 1)
                m_type = e_long_double;
            else
                error.SetErrorStringWithFormat ("'%s' is not a valid float string value", value_str);
        }
        else
        {
            error.SetErrorStringWithFormat ("unsupported float byte size: %u", byte_size);
            return error;
        }
        break;

    case eEncodingVector:
        error.SetErrorString ("vector encoding unsupported.");
        break;
    }
    if (error.Fail())
        m_type = e_void;

    return error;
}

bool
Scalar::SignExtend (uint32_t sign_bit_pos)
{
    const uint32_t max_bit_pos = GetByteSize() * 8;

    if (sign_bit_pos < max_bit_pos)
    {
        switch (m_type)
        {
        default:
        case Scalar::e_void:
        case Scalar::e_float:
        case Scalar::e_double:
        case Scalar::e_long_double: 
            return false;
            
        case Scalar::e_sint:            
        case Scalar::e_uint:
            if (max_bit_pos == sign_bit_pos)
                return true;
            else if (sign_bit_pos < (max_bit_pos-1))
            {
                unsigned int sign_bit = 1u << sign_bit_pos;
                if (m_data.uint & sign_bit)
                {
                    const unsigned int mask = ~(sign_bit) + 1u;
                    m_data.uint |= mask;
                }
                return true;
            }
            break;
            
        case Scalar::e_slong:
        case Scalar::e_ulong:
            if (max_bit_pos == sign_bit_pos)
                return true;
            else if (sign_bit_pos < (max_bit_pos-1))
            {
                unsigned long sign_bit = 1ul << sign_bit_pos;
                if (m_data.ulong & sign_bit)
                {
                    const unsigned long mask = ~(sign_bit) + 1ul;
                    m_data.ulong |= mask;
                }
                return true;
            }
            break;
            
        case Scalar::e_slonglong:
        case Scalar::e_ulonglong:
            if (max_bit_pos == sign_bit_pos)
                return true;
            else if (sign_bit_pos < (max_bit_pos-1))
            {
                unsigned long long sign_bit = 1ull << sign_bit_pos;
                if (m_data.ulonglong & sign_bit)
                {
                    const unsigned long long mask = ~(sign_bit) + 1ull;
                    m_data.ulonglong |= mask;
                }
                return true;
            }
            break;
        }
    }
    return false;
}

uint32_t
Scalar::GetAsMemoryData (void *dst,
                         uint32_t dst_len, 
                         lldb::ByteOrder dst_byte_order,
                         Error &error) const
{
    // Get a data extractor that points to the native scalar data
    DataExtractor data;
    if (!GetData(data))
    {
        error.SetErrorString ("invalid scalar value");
        return 0;
    }

    const size_t src_len = data.GetByteSize();

    // Prepare a memory buffer that contains some or all of the register value
    const uint32_t bytes_copied = data.CopyByteOrderedData (0,                  // src offset
                                                            src_len,            // src length
                                                            dst,                // dst buffer
                                                            dst_len,            // dst length
                                                            dst_byte_order);    // dst byte order
    if (bytes_copied == 0) 
        error.SetErrorString ("failed to copy data");

    return bytes_copied;
}

bool
Scalar::ExtractBitfield (uint32_t bit_size, 
                         uint32_t bit_offset)
{
    if (bit_size == 0)
        return true;

    uint32_t msbit = bit_offset + bit_size - 1;
    uint32_t lsbit = bit_offset;
    switch (m_type)
    {
        default:
        case Scalar::e_void:
            break;
            
        case e_float:
            if (sizeof(m_data.flt) == sizeof(int))
                m_data.sint = SignedBits (m_data.sint, msbit, lsbit);
            else if (sizeof(m_data.flt) == sizeof(unsigned long))
                m_data.slong = SignedBits (m_data.slong, msbit, lsbit);
            else if (sizeof(m_data.flt) == sizeof(unsigned long long))
                m_data.slonglong = SignedBits (m_data.slonglong, msbit, lsbit);
            else
                return false;
            return true;
            
        case e_double:
            if (sizeof(m_data.dbl) == sizeof(int))
                m_data.sint = SignedBits (m_data.sint, msbit, lsbit);
            else if (sizeof(m_data.dbl) == sizeof(unsigned long))
                m_data.slong = SignedBits (m_data.slong, msbit, lsbit);
            else if (sizeof(m_data.dbl) == sizeof(unsigned long long))
                m_data.slonglong = SignedBits (m_data.slonglong, msbit, lsbit);
            else
                return false;
            return true;
            
        case e_long_double:
            if (sizeof(m_data.ldbl) == sizeof(int))
                m_data.sint = SignedBits (m_data.sint, msbit, lsbit);
            else if (sizeof(m_data.ldbl) == sizeof(unsigned long))
                m_data.slong = SignedBits (m_data.slong, msbit, lsbit);
            else if (sizeof(m_data.ldbl) == sizeof(unsigned long long))
                m_data.slonglong = SignedBits (m_data.slonglong, msbit, lsbit);
            else
                return false;
            return true;
            
        case Scalar::e_sint:
            m_data.sint = SignedBits (m_data.sint, msbit, lsbit);
            return true;

        case Scalar::e_uint:
            m_data.uint = UnsignedBits (m_data.uint, msbit, lsbit);
            return true;
            
        case Scalar::e_slong:
            m_data.slong = SignedBits (m_data.slong, msbit, lsbit);
            return true;

        case Scalar::e_ulong:
            m_data.ulong = SignedBits (m_data.ulong, msbit, lsbit);
            return true;
            
        case Scalar::e_slonglong:
            m_data.slonglong = SignedBits (m_data.slonglong, msbit, lsbit);
            return true;

        case Scalar::e_ulonglong:
            m_data.ulonglong = SignedBits (m_data.ulonglong, msbit, lsbit);
            return true;
    }
    return false;
}





bool
lldb_private::operator== (const Scalar& lhs, const Scalar& rhs)
{
    // If either entry is void then we can just compare the types
    if (lhs.m_type == Scalar::e_void || rhs.m_type == Scalar::e_void)
        return lhs.m_type == rhs.m_type;

    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    switch (PromoteToMaxType(lhs, rhs, temp_value, a, b))
    {
    default:
    case Scalar::e_void:            break;
    case Scalar::e_sint:            return a->m_data.sint       == b->m_data.sint;
    case Scalar::e_uint:            return a->m_data.uint       == b->m_data.uint;
    case Scalar::e_slong:           return a->m_data.slong      == b->m_data.slong;
    case Scalar::e_ulong:           return a->m_data.ulong      == b->m_data.ulong;
    case Scalar::e_slonglong:       return a->m_data.slonglong  == b->m_data.slonglong;
    case Scalar::e_ulonglong:       return a->m_data.ulonglong  == b->m_data.ulonglong;
    case Scalar::e_float:           return a->m_data.flt        == b->m_data.flt;
    case Scalar::e_double:      return a->m_data.dbl        == b->m_data.dbl;
    case Scalar::e_long_double: return a->m_data.ldbl       == b->m_data.ldbl;
    }
    return false;
}

bool
lldb_private::operator!= (const Scalar& lhs, const Scalar& rhs)
{
    // If either entry is void then we can just compare the types
    if (lhs.m_type == Scalar::e_void || rhs.m_type == Scalar::e_void)
        return lhs.m_type != rhs.m_type;

    Scalar temp_value;  // A temp value that might get a copy of either promoted value
    const Scalar* a;
    const Scalar* b;
    switch (PromoteToMaxType(lhs, rhs, temp_value, a, b))
    {
    default:
    case Scalar::e_void:            break;
    case Scalar::e_sint:            return a->m_data.sint       != b->m_data.sint;
    case Scalar::e_uint:            return a->m_data.uint       != b->m_data.uint;
    case Scalar::e_slong:           return a->m_data.slong      != b->m_data.slong;
    case Scalar::e_ulong:           return a->m_data.ulong      != b->m_data.ulong;
    case Scalar::e_slonglong:       return a->m_data.slonglong  != b->m_data.slonglong;
    case Scalar::e_ulonglong:       return a->m_data.ulonglong  != b->m_data.ulonglong;
    case Scalar::e_float:           return a->m_data.flt        != b->m_data.flt;
    case Scalar::e_double:      return a->m_data.dbl        != b->m_data.dbl;
    case Scalar::e_long_double: return a->m_data.ldbl       != b->m_data.ldbl;
    }
    return true;
}

bool
lldb_private::operator< (const Scalar& lhs, const Scalar& rhs)
{
    if (lhs.m_type == Scalar::e_void || rhs.m_type == Scalar::e_void)
        return false;

    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    switch (PromoteToMaxType(lhs, rhs, temp_value, a, b))
    {
    default:
    case Scalar::e_void:            break;
    case Scalar::e_sint:            return a->m_data.sint       < b->m_data.sint;
    case Scalar::e_uint:            return a->m_data.uint       < b->m_data.uint;
    case Scalar::e_slong:           return a->m_data.slong      < b->m_data.slong;
    case Scalar::e_ulong:           return a->m_data.ulong      < b->m_data.ulong;
    case Scalar::e_slonglong:       return a->m_data.slonglong  < b->m_data.slonglong;
    case Scalar::e_ulonglong:       return a->m_data.ulonglong  < b->m_data.ulonglong;
    case Scalar::e_float:           return a->m_data.flt        < b->m_data.flt;
    case Scalar::e_double:      return a->m_data.dbl        < b->m_data.dbl;
    case Scalar::e_long_double: return a->m_data.ldbl       < b->m_data.ldbl;
    }
    return false;
}

bool
lldb_private::operator<= (const Scalar& lhs, const Scalar& rhs)
{
    if (lhs.m_type == Scalar::e_void || rhs.m_type == Scalar::e_void)
        return false;

    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    switch (PromoteToMaxType(lhs, rhs, temp_value, a, b))
    {
    default:
    case Scalar::e_void:            break;
    case Scalar::e_sint:            return a->m_data.sint       <= b->m_data.sint;
    case Scalar::e_uint:            return a->m_data.uint       <= b->m_data.uint;
    case Scalar::e_slong:           return a->m_data.slong      <= b->m_data.slong;
    case Scalar::e_ulong:           return a->m_data.ulong      <= b->m_data.ulong;
    case Scalar::e_slonglong:       return a->m_data.slonglong  <= b->m_data.slonglong;
    case Scalar::e_ulonglong:       return a->m_data.ulonglong  <= b->m_data.ulonglong;
    case Scalar::e_float:           return a->m_data.flt        <= b->m_data.flt;
    case Scalar::e_double:      return a->m_data.dbl        <= b->m_data.dbl;
    case Scalar::e_long_double: return a->m_data.ldbl       <= b->m_data.ldbl;
    }
    return false;
}


bool
lldb_private::operator> (const Scalar& lhs, const Scalar& rhs)
{
    if (lhs.m_type == Scalar::e_void || rhs.m_type == Scalar::e_void)
        return false;

    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    switch (PromoteToMaxType(lhs, rhs, temp_value, a, b))
    {
    default:
    case Scalar::e_void:            break;
    case Scalar::e_sint:            return a->m_data.sint       > b->m_data.sint;
    case Scalar::e_uint:            return a->m_data.uint       > b->m_data.uint;
    case Scalar::e_slong:           return a->m_data.slong      > b->m_data.slong;
    case Scalar::e_ulong:           return a->m_data.ulong      > b->m_data.ulong;
    case Scalar::e_slonglong:       return a->m_data.slonglong  > b->m_data.slonglong;
    case Scalar::e_ulonglong:       return a->m_data.ulonglong  > b->m_data.ulonglong;
    case Scalar::e_float:           return a->m_data.flt        > b->m_data.flt;
    case Scalar::e_double:      return a->m_data.dbl        > b->m_data.dbl;
    case Scalar::e_long_double: return a->m_data.ldbl       > b->m_data.ldbl;
    }
    return false;
}

bool
lldb_private::operator>= (const Scalar& lhs, const Scalar& rhs)
{
    if (lhs.m_type == Scalar::e_void || rhs.m_type == Scalar::e_void)
        return false;

    Scalar temp_value;
    const Scalar* a;
    const Scalar* b;
    switch (PromoteToMaxType(lhs, rhs, temp_value, a, b))
    {
    default:
    case Scalar::e_void:            break;
    case Scalar::e_sint:            return a->m_data.sint       >= b->m_data.sint;
    case Scalar::e_uint:            return a->m_data.uint       >= b->m_data.uint;
    case Scalar::e_slong:           return a->m_data.slong      >= b->m_data.slong;
    case Scalar::e_ulong:           return a->m_data.ulong      >= b->m_data.ulong;
    case Scalar::e_slonglong:       return a->m_data.slonglong  >= b->m_data.slonglong;
    case Scalar::e_ulonglong:       return a->m_data.ulonglong  >= b->m_data.ulonglong;
    case Scalar::e_float:           return a->m_data.flt        >= b->m_data.flt;
    case Scalar::e_double:      return a->m_data.dbl        >= b->m_data.dbl;
    case Scalar::e_long_double: return a->m_data.ldbl       >= b->m_data.ldbl;
    }
    return false;
}




