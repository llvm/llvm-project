//===-- Mangled.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#if defined(__APPLE__)
#define USE_BUILTIN_LIBCXXABI_DEMANGLER 1
#endif

#if defined(USE_BUILTIN_LIBCXXABI_DEMANGLER)
#include "lldb/Core/cxa_demangle.h"
#else
// FreeBSD9-STABLE requires this to know about size_t in cxxabi.h
#include <cstddef>
#include <cxxabi.h>
#endif


#include "llvm/ADT/DenseMap.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

using namespace lldb_private;

static inline bool
cstring_is_mangled (const char *s)
{
    if (s)
        return s[0] == '_' && s[1] == 'Z';
    return false;
}

#pragma mark Mangled
//----------------------------------------------------------------------
// Default constructor
//----------------------------------------------------------------------
Mangled::Mangled () :
    m_mangled(),
    m_demangled()
{
}

//----------------------------------------------------------------------
// Constructor with an optional string and a boolean indicating if it is
// the mangled version.
//----------------------------------------------------------------------
Mangled::Mangled (const ConstString &s, bool mangled) :
    m_mangled(),
    m_demangled()
{
    if (s)
        SetValue(s, mangled);
}

Mangled::Mangled (const ConstString &s) :
    m_mangled(),
    m_demangled()
{
    if (s)
        SetValue(s);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Mangled::~Mangled ()
{
}

//----------------------------------------------------------------------
// Convert to pointer operator. This allows code to check any Mangled
// objects to see if they contain anything valid using code such as:
//
//  Mangled mangled(...);
//  if (mangled)
//  { ...
//----------------------------------------------------------------------
Mangled::operator void* () const
{
    return (m_mangled) ? const_cast<Mangled*>(this) : NULL;
}

//----------------------------------------------------------------------
// Logical NOT operator. This allows code to check any Mangled
// objects to see if they are invalid using code such as:
//
//  Mangled mangled(...);
//  if (!file_spec)
//  { ...
//----------------------------------------------------------------------
bool
Mangled::operator! () const
{
    return !m_mangled;
}

//----------------------------------------------------------------------
// Clear the mangled and demangled values.
//----------------------------------------------------------------------
void
Mangled::Clear ()
{
    m_mangled.Clear();
    m_demangled.Clear();
}


//----------------------------------------------------------------------
// Compare the the string values.
//----------------------------------------------------------------------
int
Mangled::Compare (const Mangled& a, const Mangled& b)
{
    return ConstString::Compare(a.GetName(ePreferMangled), a.GetName(ePreferMangled));
}



//----------------------------------------------------------------------
// Set the string value in this objects. If "mangled" is true, then
// the mangled named is set with the new value in "s", else the
// demangled name is set.
//----------------------------------------------------------------------
void
Mangled::SetValue (const ConstString &s, bool mangled)
{
    if (s)
    {
        if (mangled)
        {
            m_demangled.Clear();
            m_mangled = s;
        }
        else
        {
            m_demangled = s;
            m_mangled.Clear();
        }
    }
    else
    {
        m_demangled.Clear();
        m_mangled.Clear();
    }
}

void
Mangled::SetValue (const ConstString &name)
{
    if (name)
    {
        if (cstring_is_mangled(name.GetCString()))
        {
            m_demangled.Clear();
            m_mangled = name;
        }
        else
        {
            m_demangled = name;
            m_mangled.Clear();
        }
    }
    else
    {
        m_demangled.Clear();
        m_mangled.Clear();
    }
}


//----------------------------------------------------------------------
// Generate the demangled name on demand using this accessor. Code in
// this class will need to use this accessor if it wishes to decode
// the demangled name. The result is cached and will be kept until a
// new string value is supplied to this object, or until the end of the
// object's lifetime.
//----------------------------------------------------------------------
const ConstString&
Mangled::GetDemangledName () const
{
    // Check to make sure we have a valid mangled name and that we
    // haven't already decoded our mangled name.
    if (m_mangled && !m_demangled)
    {
        // We need to generate and cache the demangled name.
        Timer scoped_timer (__PRETTY_FUNCTION__,
                            "Mangled::GetDemangledName (m_mangled = %s)",
                            m_mangled.GetCString());

        // Don't bother running anything that isn't mangled
        const char *mangled_cstr = m_mangled.GetCString();
        if (cstring_is_mangled(mangled_cstr))
        {
            if (!m_mangled.GetMangledCounterpart(m_demangled))
            {
                // We didn't already mangle this name, demangle it and if all goes well
                // add it to our map.
#if defined(USE_BUILTIN_LIBCXXABI_DEMANGLER)
                char *demangled_name = lldb_cxxabiv1::__cxa_demangle (mangled_cstr, NULL, NULL, NULL);
#else
                char *demangled_name = abi::__cxa_demangle (mangled_cstr, NULL, NULL, NULL);
#endif

                if (demangled_name)
                {
                    m_demangled.SetCStringWithMangledCounterpart(demangled_name, m_mangled);                    
                    free (demangled_name);
                }
            }
        }
        if (!m_demangled)
        {
            // Set the demangled string to the empty string to indicate we
            // tried to parse it once and failed.
            m_demangled.SetCString("");
        }
    }

    return m_demangled;
}


bool
Mangled::NameMatches (const RegularExpression& regex) const
{
    if (m_mangled && regex.Execute (m_mangled.AsCString()))
        return true;
    
    if (GetDemangledName() && regex.Execute (m_demangled.AsCString()))
        return true;
    return false;
}

//----------------------------------------------------------------------
// Get the demangled name if there is one, else return the mangled name.
//----------------------------------------------------------------------
const ConstString&
Mangled::GetName (Mangled::NamePreference preference) const
{
    if (preference == ePreferDemangled)
    {
        // Call the accessor to make sure we get a demangled name in case
        // it hasn't been demangled yet...
        if (GetDemangledName())
            return m_demangled;
        return m_mangled;
    }
    else
    {
        if (m_mangled)
            return m_mangled;
        return GetDemangledName();
    }
}

//----------------------------------------------------------------------
// Dump a Mangled object to stream "s". We don't force our
// demangled name to be computed currently (we don't use the accessor).
//----------------------------------------------------------------------
void
Mangled::Dump (Stream *s) const
{
    if (m_mangled)
    {
        *s << ", mangled = " << m_mangled;
    }
    if (m_demangled)
    {
        const char * demangled = m_demangled.AsCString();
        s->Printf(", demangled = %s", demangled[0] ? demangled : "<error>");
    }
}

//----------------------------------------------------------------------
// Dumps a debug version of this string with extra object and state
// information to stream "s".
//----------------------------------------------------------------------
void
Mangled::DumpDebug (Stream *s) const
{
    s->Printf("%*p: Mangled mangled = ", (int)sizeof(void*) * 2, this);
    m_mangled.DumpDebug(s);
    s->Printf(", demangled = ");
    m_demangled.DumpDebug(s);
}

//----------------------------------------------------------------------
// Return the size in byte that this object takes in memory. The size
// includes the size of the objects it owns, and not the strings that
// it references because they are shared strings.
//----------------------------------------------------------------------
size_t
Mangled::MemorySize () const
{
    return m_mangled.MemorySize() + m_demangled.MemorySize();
}

//----------------------------------------------------------------------
// Dump OBJ to the supplied stream S.
//----------------------------------------------------------------------
Stream&
operator << (Stream& s, const Mangled& obj)
{
    if (obj.GetMangledName())
        s << "mangled = '" << obj.GetMangledName() << "'";

    const ConstString& demangled = obj.GetDemangledName();
    if (demangled)
        s << ", demangled = '" << demangled << '\'';
    else
        s << ", demangled = <error>";
    return s;
}
