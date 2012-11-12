//===-- ConstString.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Stream.h"
#include "lldb/Host/Mutex.h"
#include "llvm/ADT/StringMap.h"

using namespace lldb_private;


class Pool
{
public:
    typedef const char * StringPoolValueType;
    typedef llvm::StringMap<StringPoolValueType, llvm::BumpPtrAllocator> StringPool;
    typedef llvm::StringMapEntry<StringPoolValueType> StringPoolEntryType;
    
    //------------------------------------------------------------------
    // Default constructor
    //
    // Initialize the member variables and create the empty string.
    //------------------------------------------------------------------
    Pool () :
        m_mutex (Mutex::eMutexTypeRecursive),
        m_string_map ()
    {
    }

    //------------------------------------------------------------------
    // Destructor
    //------------------------------------------------------------------
    ~Pool ()
    {
    }


    static StringPoolEntryType &
    GetStringMapEntryFromKeyData (const char *keyData)
    {
        char *ptr = const_cast<char*>(keyData) - sizeof (StringPoolEntryType);
        return *reinterpret_cast<StringPoolEntryType*>(ptr);
    }

    size_t
    GetConstCStringLength (const char *ccstr) const
    {
        if (ccstr)
        {
            const StringPoolEntryType&entry = GetStringMapEntryFromKeyData (ccstr);
            return entry.getKey().size();
        }
        return 0;
    }

    StringPoolValueType
    GetMangledCounterpart (const char *ccstr) const
    {
        if (ccstr)
            return GetStringMapEntryFromKeyData (ccstr).getValue();
        return 0;
    }

    bool
    SetMangledCounterparts (const char *key_ccstr, const char *value_ccstr)
    {
        if (key_ccstr && value_ccstr)
        {
            GetStringMapEntryFromKeyData (key_ccstr).setValue(value_ccstr);
            GetStringMapEntryFromKeyData (value_ccstr).setValue(key_ccstr);
            return true;
        }
        return false;
    }

    const char *
    GetConstCString (const char *cstr)
    {
        if (cstr)
            return GetConstCStringWithLength (cstr, strlen (cstr));
        return NULL;
    }

    const char *
    GetConstCStringWithLength (const char *cstr, int cstr_len)
    {
        if (cstr)
        {
            Mutex::Locker locker (m_mutex);
            llvm::StringRef string_ref (cstr, cstr_len);
            StringPoolEntryType& entry = m_string_map.GetOrCreateValue (string_ref, (StringPoolValueType)NULL);
            return entry.getKeyData();
        }
        return NULL;
    }

    const char *
    GetConstCStringWithStringRef (const llvm::StringRef &string_ref)
    {
        if (string_ref.data())
        {
            Mutex::Locker locker (m_mutex);
            StringPoolEntryType& entry = m_string_map.GetOrCreateValue (string_ref, (StringPoolValueType)NULL);
            return entry.getKeyData();
        }
        return NULL;
    }

    const char *
    GetConstCStringAndSetMangledCounterPart (const char *demangled_cstr, const char *mangled_ccstr)
    {
        if (demangled_cstr)
        {
            Mutex::Locker locker (m_mutex);
            // Make string pool entry with the mangled counterpart already set
            StringPoolEntryType& entry = m_string_map.GetOrCreateValue (llvm::StringRef (demangled_cstr), mangled_ccstr);

            // Extract the const version of the demangled_cstr
            const char *demangled_ccstr = entry.getKeyData();
            // Now assign the demangled const string as the counterpart of the
            // mangled const string...
            GetStringMapEntryFromKeyData (mangled_ccstr).setValue(demangled_ccstr);
            // Return the constant demangled C string
            return demangled_ccstr;
        }
        return NULL;
    }

    const char *
    GetConstTrimmedCStringWithLength (const char *cstr, int cstr_len)
    {
        if (cstr)
        {
            int trimmed_len = std::min<int> (strlen (cstr), cstr_len);
            return GetConstCStringWithLength (cstr, trimmed_len);
        }
        return NULL;
    }

    //------------------------------------------------------------------
    // Return the size in bytes that this object and any items in its
    // collection of uniqued strings + data count values takes in
    // memory.
    //------------------------------------------------------------------
    size_t
    MemorySize() const
    {
        Mutex::Locker locker (m_mutex);
        size_t mem_size = sizeof(Pool);
        const_iterator end = m_string_map.end();
        for (const_iterator pos = m_string_map.begin(); pos != end; ++pos)
        {
            mem_size += sizeof(StringPoolEntryType) + pos->getKey().size();
        }
        return mem_size;
    }

protected:
    //------------------------------------------------------------------
    // Typedefs
    //------------------------------------------------------------------
    typedef StringPool::iterator iterator;
    typedef StringPool::const_iterator const_iterator;

    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    mutable Mutex m_mutex;
    StringPool m_string_map;
};

//----------------------------------------------------------------------
// Frameworks and dylibs aren't supposed to have global C++
// initializers so we hide the string pool in a static function so
// that it will get initialized on the first call to this static
// function.
//
// Note, for now we make the string pool a pointer to the pool, because
// we can't guarantee that some objects won't get destroyed after the
// global destructor chain is run, and trying to make sure no destructors
// touch ConstStrings is difficult.  So we leak the pool instead.
//
// FIXME: If we are going to keep it this way we should come up with some
// abstraction to "pthread_once" so we don't have to check the pointer
// every time.
//----------------------------------------------------------------------
static Pool &
StringPool()
{
    static Mutex g_pool_initialization_mutex;
    static Pool *g_string_pool = NULL;

    if (g_string_pool == NULL)
    {
        Mutex::Locker initialization_locker(g_pool_initialization_mutex);
        if (g_string_pool == NULL)
        {
            g_string_pool = new Pool();
        }
    }
    
    return *g_string_pool;
}

ConstString::ConstString (const char *cstr) :
    m_string (StringPool().GetConstCString (cstr))
{
}

ConstString::ConstString (const char *cstr, size_t cstr_len) :
    m_string (StringPool().GetConstCStringWithLength (cstr, cstr_len))
{
}

ConstString::ConstString (const llvm::StringRef &s) :
    m_string (StringPool().GetConstCStringWithLength (s.data(), s.size()))
{
}

bool
ConstString::operator < (const ConstString& rhs) const
{
    if (m_string == rhs.m_string)
        return false;

    llvm::StringRef lhs_string_ref (m_string, StringPool().GetConstCStringLength (m_string));
    llvm::StringRef rhs_string_ref (rhs.m_string, StringPool().GetConstCStringLength (rhs.m_string));

    // If both have valid C strings, then return the comparison
    if (lhs_string_ref.data() && rhs_string_ref.data())
        return lhs_string_ref < rhs_string_ref;

    // Else one of them was NULL, so if LHS is NULL then it is less than
    return lhs_string_ref.data() == NULL;
}

Stream&
lldb_private::operator << (Stream& s, const ConstString& str)
{
    const char *cstr = str.GetCString();
    if (cstr)
        s << cstr;

    return s;
}

size_t
ConstString::GetLength () const
{
    return StringPool().GetConstCStringLength (m_string);
}

int
ConstString::Compare (const ConstString& lhs, const ConstString& rhs)
{
    // If the iterators are the same, this is the same string
    register const char *lhs_cstr = lhs.m_string;
    register const char *rhs_cstr = rhs.m_string;
    if (lhs_cstr == rhs_cstr)
        return 0;
    if (lhs_cstr && rhs_cstr)
    {
        llvm::StringRef lhs_string_ref (lhs_cstr, StringPool().GetConstCStringLength (lhs_cstr));
        llvm::StringRef rhs_string_ref (rhs_cstr, StringPool().GetConstCStringLength (rhs_cstr));
        return lhs_string_ref.compare(rhs_string_ref);
    }

    if (lhs_cstr)
        return +1;  // LHS isn't NULL but RHS is
    else
        return -1;  // LHS is NULL but RHS isn't
}

void
ConstString::Dump(Stream *s, const char *fail_value) const
{
    if (s)
    {
        const char *cstr = AsCString (fail_value);
        if (cstr)
            s->PutCString (cstr);
    }
}

void
ConstString::DumpDebug(Stream *s) const
{
    const char *cstr = GetCString ();
    size_t cstr_len = GetLength();
    // Only print the parens if we have a non-NULL string
    const char *parens = cstr ? "\"" : "";
    s->Printf("%*p: ConstString, string = %s%s%s, length = %llu", (int)sizeof(void*) * 2, this, parens, cstr, parens, (uint64_t)cstr_len);
}

void
ConstString::SetCString (const char *cstr)
{
    m_string = StringPool().GetConstCString (cstr);
}

void
ConstString::SetString (const llvm::StringRef &s)
{
    m_string = StringPool().GetConstCStringWithLength (s.data(), s.size());
}

void
ConstString::SetCStringWithMangledCounterpart (const char *demangled, const ConstString &mangled)
{
    m_string = StringPool().GetConstCStringAndSetMangledCounterPart (demangled, mangled.m_string);
}

bool
ConstString::GetMangledCounterpart (ConstString &counterpart) const
{
    counterpart.m_string = StringPool().GetMangledCounterpart(m_string);
    return counterpart;
}

void
ConstString::SetCStringWithLength (const char *cstr, size_t cstr_len)
{
    m_string = StringPool().GetConstCStringWithLength(cstr, cstr_len);
}

void
ConstString::SetTrimmedCStringWithLength (const char *cstr, size_t cstr_len)
{
    m_string = StringPool().GetConstTrimmedCStringWithLength (cstr, cstr_len);
}

size_t
ConstString::StaticMemorySize()
{
    // Get the size of the static string pool
    return StringPool().MemorySize();
}
