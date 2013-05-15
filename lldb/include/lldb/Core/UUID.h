//===-- UUID.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_UUID_h_
#define liblldb_UUID_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private.h"

namespace lldb_private {

class UUID
{
public:
    typedef uint8_t ValueType[16];

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    UUID ();
    UUID (const UUID& rhs);
    UUID (const void *uuid_bytes, uint32_t num_uuid_bytes);

    ~UUID ();

    const UUID&
    operator=(const UUID& rhs);

    void
    Clear ();

    void
    Dump (Stream *s) const;

    const void *
    GetBytes() const;

    static size_t
    GetByteSize();

    bool
    IsValid () const;

    void
    SetBytes (const void *uuid_bytes);

    std::string
    GetAsString () const;

    size_t
    SetFromCString (const char *c_str);

    // Decode as many UUID bytes (up to 16) as possible from the C string "cstr"
    // This is used for auto completion where a partial UUID might have been
    // typed in. It 
    //------------------------------------------------------------------
    /// Decode as many UUID bytes (up to 16) as possible from the C
    /// string \a cstr.
    ///
    /// @param[in] cstr
    ///     A NULL terminate C string that points at a UUID string value
    ///     (no leading spaces). The string must contain only hex
    ///     characters and optionally can contain the '-' sepearators.
    ///
    /// @param[in] uuid_bytes
    ///     A buffer of bytes that will contain a full or patially
    ///     decoded UUID.
    ///
    /// @param[out] end
    ///     If \a end is not NULL, it will be filled in with the a
    ///     pointer to the character after the last successfully decoded
    ///     byte.
    ///
    /// @return
    ///     Returns the number of bytes that were successfully decoded
    ///     which should be 16 if a full UUID value was properly decoded.
    //------------------------------------------------------------------
    static size_t
    DecodeUUIDBytesFromCString (const char *cstr, ValueType &uuid_bytes, const char **end);
    
protected:
    //------------------------------------------------------------------
    // Classes that inherit from UUID can see and modify these
    //------------------------------------------------------------------
    ValueType m_uuid;
};

bool operator == (const UUID &lhs, const UUID &rhs);
bool operator != (const UUID &lhs, const UUID &rhs);
bool operator <  (const UUID &lhs, const UUID &rhs);
bool operator <= (const UUID &lhs, const UUID &rhs);
bool operator >  (const UUID &lhs, const UUID &rhs);
bool operator >= (const UUID &lhs, const UUID &rhs);

} // namespace lldb_private

#endif  // liblldb_UUID_h_
