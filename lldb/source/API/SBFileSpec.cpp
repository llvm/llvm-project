//===-- SBFileSpec.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <limits.h>

#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBStream.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"

using namespace lldb;
using namespace lldb_private;



SBFileSpec::SBFileSpec () :
    m_opaque_ap()
{
}

SBFileSpec::SBFileSpec (const SBFileSpec &rhs) :
    m_opaque_ap()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (rhs.m_opaque_ap.get())
        m_opaque_ap.reset (new FileSpec (rhs.get()));

    if (log)
    {
        SBStream sstr;
        GetDescription (sstr);
        log->Printf ("SBFileSpec::SBFileSpec (const SBFileSpec rhs.ap=%p) => SBFileSpec(%p): %s",
                     rhs.m_opaque_ap.get(), m_opaque_ap.get(), sstr.GetData());
    }
}

// Deprected!!!
SBFileSpec::SBFileSpec (const char *path) :
    m_opaque_ap(new FileSpec (path, true))
{
}

SBFileSpec::SBFileSpec (const char *path, bool resolve) :
    m_opaque_ap(new FileSpec (path, resolve))
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBFileSpec::SBFileSpec (path=\"%s\", resolve=%i) => SBFileSpec(%p)", path, 
                     resolve, m_opaque_ap.get());
}

SBFileSpec::~SBFileSpec ()
{
}

const SBFileSpec &
SBFileSpec::operator = (const SBFileSpec &rhs)
{
    if (this != &rhs)
    {
        if (rhs.IsValid())
            m_opaque_ap.reset (new lldb_private::FileSpec(*rhs.m_opaque_ap.get()));
    }
    return *this;
}

bool
SBFileSpec::IsValid() const
{
    return m_opaque_ap.get() != NULL;
}

bool
SBFileSpec::Exists () const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    bool result = false;
    if (m_opaque_ap.get())
        result = m_opaque_ap->Exists();

    if (log)
        log->Printf ("SBFileSpec(%p)::Exists () => %s", m_opaque_ap.get(), (result ? "true" : "false"));

    return result;
}

bool
SBFileSpec::ResolveExecutableLocation ()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->ResolveExecutableLocation ();
    return false;
}

int
SBFileSpec::ResolvePath (const char *src_path, char *dst_path, size_t dst_len)
{
    return lldb_private::FileSpec::Resolve (src_path, dst_path, dst_len);
}

const char *
SBFileSpec::GetFilename() const
{
    const char *s = NULL;
    if (m_opaque_ap.get())
        s = m_opaque_ap->GetFilename().AsCString();

    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (s)
            log->Printf ("SBFileSpec(%p)::GetFilename () => \"%s\"", m_opaque_ap.get(), s);
        else
            log->Printf ("SBFileSpec(%p)::GetFilename () => NULL", m_opaque_ap.get());
    }

    return s;
}

const char *
SBFileSpec::GetDirectory() const
{
    const char *s = NULL;
    if (m_opaque_ap.get())
        s = m_opaque_ap->GetDirectory().AsCString();
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (s)
            log->Printf ("SBFileSpec(%p)::GetDirectory () => \"%s\"", m_opaque_ap.get(), s);
        else
            log->Printf ("SBFileSpec(%p)::GetDirectory () => NULL", m_opaque_ap.get());
    }
    return s;
}

uint32_t
SBFileSpec::GetPath (char *dst_path, size_t dst_len) const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    uint32_t result = 0;
    if (m_opaque_ap.get())
        result = m_opaque_ap->GetPath (dst_path, dst_len);

    if (log)
        log->Printf ("SBFileSpec(%p)::GetPath (dst_path=\"%.*s\", dst_len=%" PRIu64 ") => %u",
                     m_opaque_ap.get(), result, dst_path, (uint64_t)dst_len, result);

    if (result == 0 && dst_path && dst_len > 0)
        *dst_path = '\0';
    return result;
}


const lldb_private::FileSpec *
SBFileSpec::operator->() const
{
    return m_opaque_ap.get();
}

const lldb_private::FileSpec *
SBFileSpec::get() const
{
    return m_opaque_ap.get();
}


const lldb_private::FileSpec &
SBFileSpec::operator*() const
{
    return *m_opaque_ap.get();
}

const lldb_private::FileSpec &
SBFileSpec::ref() const
{
    return *m_opaque_ap.get();
}


void
SBFileSpec::SetFileSpec (const lldb_private::FileSpec& fs)
{
    if (m_opaque_ap.get())
        *m_opaque_ap = fs;
    else
        m_opaque_ap.reset (new FileSpec (fs));
}

bool
SBFileSpec::GetDescription (SBStream &description) const
{
    Stream &strm = description.ref();
    if (m_opaque_ap.get())
    {
        char path[PATH_MAX];
        if (m_opaque_ap->GetPath(path, sizeof(path)))
            strm.PutCString (path);
    }
    else
        strm.PutCString ("No value");
    
    return true;
}
