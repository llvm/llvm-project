//===-- UUID.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/UUID.h"
// C Includes
#include <string.h>
#include <stdio.h>
#include <ctype.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"

namespace lldb_private {

UUID::UUID()
{
    ::memset (m_uuid, 0, sizeof(m_uuid));
}

UUID::UUID(const UUID& rhs)
{
    ::memcpy (m_uuid, rhs.m_uuid, sizeof (m_uuid));
}

UUID::UUID (const void *uuid_bytes, uint32_t num_uuid_bytes)
{
    if (uuid_bytes && num_uuid_bytes >= 16)
        ::memcpy (m_uuid, uuid_bytes, sizeof (m_uuid));
    else
        ::memset (m_uuid, 0, sizeof(m_uuid));
}

const UUID&
UUID::operator=(const UUID& rhs)
{
    if (this != &rhs)
        ::memcpy (m_uuid, rhs.m_uuid, sizeof (m_uuid));
    return *this;
}

UUID::~UUID()
{
}

void
UUID::Clear()
{
    ::memset (m_uuid, 0, sizeof(m_uuid));
}

const void *
UUID::GetBytes() const
{
    return m_uuid;
}

char *
UUID::GetAsCString (char *dst, size_t dst_len) const
{
    const uint8_t *u = (const uint8_t *)GetBytes();
    if (dst_len > snprintf (dst,
                            dst_len,
                            "%2.2X%2.2X%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X",
                            u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]))
        return dst;
    return NULL;
}

void
UUID::Dump (Stream *s) const
{
    const uint8_t *u = (const uint8_t *)GetBytes();
    s->Printf ("%2.2X%2.2X%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X",
              u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
}

void
UUID::SetBytes (const void *uuid_bytes)
{
    if (uuid_bytes)
        ::memcpy (m_uuid, uuid_bytes, sizeof (m_uuid));
    else
        ::memset (m_uuid, 0, sizeof(m_uuid));
}

size_t
UUID::GetByteSize()
{
    return sizeof(UUID::ValueType);
}

bool
UUID::IsValid () const
{
    return  m_uuid[0]  ||
            m_uuid[1]  ||
            m_uuid[2]  ||
            m_uuid[3]  ||
            m_uuid[4]  ||
            m_uuid[5]  ||
            m_uuid[6]  ||
            m_uuid[7]  ||
            m_uuid[8]  ||
            m_uuid[9]  ||
            m_uuid[10] ||
            m_uuid[11] ||
            m_uuid[12] ||
            m_uuid[13] ||
            m_uuid[14] ||
            m_uuid[15];
}

static inline int
xdigit_to_int (char ch)
{
    ch = tolower(ch);
    if (ch >= 'a' && ch <= 'f')
        return 10 + ch - 'a';
    return ch - '0';
}

size_t
UUID::DecodeUUIDBytesFromCString (const char *p, ValueType &uuid_bytes, const char **end)
{
    size_t uuid_byte_idx = 0;
    if (p)
    {
        while (*p)
        {
            if (isxdigit(p[0]) && isxdigit(p[1]))
            {
                int hi_nibble = xdigit_to_int(p[0]);
                int lo_nibble = xdigit_to_int(p[1]);
                // Translate the two hex nibble characters into a byte
                uuid_bytes[uuid_byte_idx] = (hi_nibble << 4) + lo_nibble;
                
                // Skip both hex digits
                p += 2;
                
                // Increment the byte that we are decoding within the UUID value
                // and break out if we are done
                if (++uuid_byte_idx == 16)
                    break;
            }
            else if (*p == '-')
            {
                // Skip dashes
                p++;
            }
            else
            {
                // UUID values can only consist of hex characters and '-' chars
                break;
            }
        }
    }
    if (end)
        *end = p;
    return uuid_byte_idx;
}
size_t
UUID::SetFromCString (const char *cstr)
{
    if (cstr == NULL)
        return 0;

    const char *p = cstr;

    // Skip leading whitespace characters
    while (isspace(*p))
        ++p;
    
    const uint32_t uuid_byte_idx = UUID::DecodeUUIDBytesFromCString (p, m_uuid, &p);

    // If we successfully decoded a UUID, return the amount of characters that
    // were consumed
    if (uuid_byte_idx == 16)
        return p - cstr;

    // Else return zero to indicate we were not able to parse a UUID value
    return 0;
}

}

bool
lldb_private::operator == (const lldb_private::UUID &lhs, const lldb_private::UUID &rhs)
{
    return ::memcmp (lhs.GetBytes(), rhs.GetBytes(), lldb_private::UUID::GetByteSize()) == 0;
}

bool
lldb_private::operator != (const lldb_private::UUID &lhs, const lldb_private::UUID &rhs)
{
    return ::memcmp (lhs.GetBytes(), rhs.GetBytes(), lldb_private::UUID::GetByteSize()) != 0;
}

bool
lldb_private::operator <  (const lldb_private::UUID &lhs, const lldb_private::UUID &rhs)
{
    return ::memcmp (lhs.GetBytes(), rhs.GetBytes(), lldb_private::UUID::GetByteSize()) <  0;
}

bool
lldb_private::operator <= (const lldb_private::UUID &lhs, const lldb_private::UUID &rhs)
{
    return ::memcmp (lhs.GetBytes(), rhs.GetBytes(), lldb_private::UUID::GetByteSize()) <= 0;
}

bool
lldb_private::operator >  (const lldb_private::UUID &lhs, const lldb_private::UUID &rhs)
{
    return ::memcmp (lhs.GetBytes(), rhs.GetBytes(), lldb_private::UUID::GetByteSize()) >  0;
}

bool
lldb_private::operator >= (const lldb_private::UUID &lhs, const lldb_private::UUID &rhs)
{
    return ::memcmp (lhs.GetBytes(), rhs.GetBytes(), lldb_private::UUID::GetByteSize()) >= 0;
}
