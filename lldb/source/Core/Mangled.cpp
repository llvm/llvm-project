//===-- Mangled.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// FreeBSD9-STABLE requires this to know about size_t in cxxabi.h
#include <cstddef>
#if defined(_MSC_VER)
#include "lldb/Host/windows/windows.h"
#include <Dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#define LLDB_USE_BUILTIN_DEMANGLER
#elif defined (__FreeBSD__)
#define LLDB_USE_BUILTIN_DEMANGLER
#else
#include <cxxabi.h>
#endif

#ifdef LLDB_USE_BUILTIN_DEMANGLER

// Provide a fast-path demangler implemented in FastDemangle.cpp until it can
// replace the existing C++ demangler with a complete implementation
#include "lldb/Core/FastDemangle.h"
#include "lldb/Core/CxaDemangle.h"

#endif


#include "llvm/ADT/DenseMap.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Logging.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"
#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include <ctype.h>
#include <string.h>
#include <stdlib.h>


using namespace lldb_private;

static inline Mangled::ManglingScheme
cstring_mangling_scheme(const char *s)
{
    if (s)
    {
        if (s[0] == '?')
            return Mangled::eManglingSchemeMSVC;
        if (s[0] == '_' && s[1] == 'Z')
            return Mangled::eManglingSchemeItanium;
    }
    return Mangled::eManglingSchemeNone;
}

static inline bool
cstring_is_mangled(const char *s)
{
    return cstring_mangling_scheme(s) != Mangled::eManglingSchemeNone;
}

static const ConstString &
get_demangled_name_without_arguments (ConstString mangled, ConstString demangled)
{
    // This pair is <mangled name, demangled name without function arguments>
    static std::pair<ConstString, ConstString> g_most_recent_mangled_to_name_sans_args;

    // Need to have the mangled & demangled names we're currently examining as statics
    // so we can return a const ref to them at the end of the func if we don't have
    // anything better.
    static ConstString g_last_mangled;
    static ConstString g_last_demangled;

    if (mangled && g_most_recent_mangled_to_name_sans_args.first == mangled)
    {
        return g_most_recent_mangled_to_name_sans_args.second;
    }

    g_last_demangled = demangled;
    g_last_mangled = mangled;

    const char *mangled_name_cstr = mangled.GetCString();

    if (demangled && mangled_name_cstr && mangled_name_cstr[0])
    {
        if (mangled_name_cstr[0] == '_' && mangled_name_cstr[1] == 'Z' &&
            (mangled_name_cstr[2] != 'T' && // avoid virtual table, VTT structure, typeinfo structure, and typeinfo mangled_name
            mangled_name_cstr[2] != 'G' && // avoid guard variables
            mangled_name_cstr[2] != 'Z'))  // named local entities (if we eventually handle eSymbolTypeData, we will want this back)
        {
            CPlusPlusLanguage::MethodName cxx_method (demangled);
            if (!cxx_method.GetBasename().empty())
            {
                std::string shortname;
                if (!cxx_method.GetContext().empty())
                    shortname = cxx_method.GetContext().str() + "::";
                shortname += cxx_method.GetBasename().str();
                ConstString result(shortname.c_str());
                g_most_recent_mangled_to_name_sans_args.first = mangled;
                g_most_recent_mangled_to_name_sans_args.second = result;
                return g_most_recent_mangled_to_name_sans_args.second;
            }
        }
    }

    if (demangled)
        return g_last_demangled;
    return g_last_mangled;
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
// Compare the string values.
//----------------------------------------------------------------------
int
Mangled::Compare (const Mangled& a, const Mangled& b)
{
    return ConstString::Compare(a.GetName(lldb::eLanguageTypeUnknown, ePreferMangled), a.GetName(lldb::eLanguageTypeUnknown, ePreferMangled));
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
Mangled::GetDemangledName (lldb::LanguageType language) const
{
    // Check to make sure we have a valid mangled name and that we
    // haven't already decoded our mangled name.
    if (m_mangled && !m_demangled)
    {
        // We need to generate and cache the demangled name.
        Timer scoped_timer (__PRETTY_FUNCTION__,
                            "Mangled::GetDemangledName (m_mangled = %s)",
                            m_mangled.GetCString());

        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_DEMANGLE);

        // Don't bother running anything that isn't mangled
        const char *mangled_name = m_mangled.GetCString();
        ManglingScheme mangling_scheme{cstring_mangling_scheme(mangled_name)};
        if (mangling_scheme != eManglingSchemeNone &&
            !m_mangled.GetMangledCounterpart(m_demangled))
        {
            // We didn't already mangle this name, demangle it and if all goes well
            // add it to our map.
            char *demangled_name = nullptr;
            switch (mangling_scheme)
            {
                case eManglingSchemeMSVC:
                {
#if defined(_MSC_VER)
                    if (log)
                        log->Printf("demangle msvc: %s", mangled_name);
                    const size_t demangled_length = 2048;
                    demangled_name = static_cast<char *>(::malloc(demangled_length));
                    ::ZeroMemory(demangled_name, demangled_length);
                    DWORD result = ::UnDecorateSymbolName(mangled_name, demangled_name, demangled_length,
                            UNDNAME_NO_ACCESS_SPECIFIERS   | // Strip public, private, protected keywords
                            UNDNAME_NO_ALLOCATION_LANGUAGE | // Strip __thiscall, __stdcall, etc keywords
                            UNDNAME_NO_THROW_SIGNATURES    | // Strip throw() specifications
                            UNDNAME_NO_MEMBER_TYPE         | // Strip virtual, static, etc specifiers
                            UNDNAME_NO_MS_KEYWORDS           // Strip all MS extension keywords
                        );
                    if (log)
                    {
                        if (demangled_name && demangled_name[0])
                            log->Printf("demangled msvc: %s -> \"%s\"", mangled_name, demangled_name);
                        else
                            log->Printf("demangled msvc: %s -> error: 0x%" PRIx64, mangled_name, result);
                    }

                    if (result == 0)
                    {
                        free(demangled_name);
                        demangled_name = nullptr;
                    }
#endif
                    break;
                }
                case eManglingSchemeItanium:
                {
#ifdef LLDB_USE_BUILTIN_DEMANGLER
                    if (log)
                        log->Printf("demangle itanium: %s", mangled_name);
                    // Try to use the fast-path demangler first for the
                    // performance win, falling back to the full demangler only
                    // when necessary
                    demangled_name = FastDemangle(mangled_name, m_mangled.GetLength());
                    if (!demangled_name)
                        demangled_name = __cxa_demangle(mangled_name, NULL, NULL, NULL);
#else
                    demangled_name = abi::__cxa_demangle(mangled_name, NULL, NULL, NULL);
#endif
                    if (log)
                    {
                        if (demangled_name)
                            log->Printf("demangled itanium: %s -> \"%s\"", mangled_name, demangled_name);
                        else
                            log->Printf("demangled itanium: %s -> error: failed to demangle", mangled_name);
                    }
                    break;
                }
                case eManglingSchemeNone:
                    break;
            }
            if (demangled_name)
            {
                m_demangled.SetCStringWithMangledCounterpart(demangled_name, m_mangled);
                free(demangled_name);
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


ConstString
Mangled::GetDisplayDemangledName (lldb::LanguageType language) const
{
    return GetDemangledName(language);
}

bool
Mangled::NameMatches (const RegularExpression& regex, lldb::LanguageType language) const
{
    if (m_mangled && regex.Execute (m_mangled.AsCString()))
        return true;

    ConstString demangled = GetDemangledName(language);
    if (demangled && regex.Execute (demangled.AsCString()))
        return true;
    return false;
}

//----------------------------------------------------------------------
// Get the demangled name if there is one, else return the mangled name.
//----------------------------------------------------------------------
ConstString
Mangled::GetName (lldb::LanguageType language, Mangled::NamePreference preference) const
{
    if (preference == ePreferMangled && m_mangled)
        return m_mangled;

    ConstString demangled = GetDemangledName(language);

    if (preference == ePreferDemangledWithoutArguments)
    {
        return get_demangled_name_without_arguments (m_mangled, demangled);
    }
    if (preference == ePreferDemangled)
    {
        // Call the accessor to make sure we get a demangled name in case
        // it hasn't been demangled yet...
        if (demangled)
            return demangled;
        return m_mangled;
    }
    return demangled;
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
    s->Printf("%*p: Mangled mangled = ", static_cast<int>(sizeof(void*) * 2),
              static_cast<const void*>(this));
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
// We "guess" the language because we can't determine a symbol's language
// from it's name.  For example, a Pascal symbol can be mangled using the
// C++ Itanium scheme, and defined in a compilation unit within the same
// module as other C++ units.  In addition, different targets could have
// different ways of mangling names from a given language, likewise the
// compilation units within those targets.
//----------------------------------------------------------------------
lldb::LanguageType
Mangled::GuessLanguage () const
{
    ConstString mangled = GetMangledName();
    if (mangled)
    {
        if (GetDemangledName(lldb::eLanguageTypeUnknown))
        {
            const char *mangled_name = mangled.GetCString();
            if (CPlusPlusLanguage::IsCPPMangledName(mangled_name))
                return lldb::eLanguageTypeC_plus_plus;
            else if (ObjCLanguage::IsPossibleObjCMethodName(mangled_name))
                return lldb::eLanguageTypeObjC;
        }
    }
    return lldb::eLanguageTypeUnknown;
}

//----------------------------------------------------------------------
// Dump OBJ to the supplied stream S.
//----------------------------------------------------------------------
Stream&
operator << (Stream& s, const Mangled& obj)
{
    if (obj.GetMangledName())
        s << "mangled = '" << obj.GetMangledName() << "'";

    const ConstString& demangled = obj.GetDemangledName(lldb::eLanguageTypeUnknown);
    if (demangled)
        s << ", demangled = '" << demangled << '\'';
    else
        s << ", demangled = <error>";
    return s;
}
