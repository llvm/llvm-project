//===-- SBCommandReturnObject.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBStream.h"

#include "lldb/Core/Log.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

SBCommandReturnObject::SBCommandReturnObject () :
    m_opaque_ap (new CommandReturnObject ())
{
}

SBCommandReturnObject::SBCommandReturnObject (const SBCommandReturnObject &rhs):
    m_opaque_ap ()
{
    if (rhs.m_opaque_ap.get())
        m_opaque_ap.reset (new CommandReturnObject (*rhs.m_opaque_ap));
}

SBCommandReturnObject::SBCommandReturnObject (CommandReturnObject *ptr) :
    m_opaque_ap (ptr)
{
}

CommandReturnObject *
SBCommandReturnObject::Release ()
{
    return m_opaque_ap.release();
}

const SBCommandReturnObject &
SBCommandReturnObject::operator = (const SBCommandReturnObject &rhs)
{
    if (this != &rhs)
    {
        if (rhs.m_opaque_ap.get())
            m_opaque_ap.reset (new CommandReturnObject (*rhs.m_opaque_ap));
        else
            m_opaque_ap.reset();
    }
    return *this;
}


SBCommandReturnObject::~SBCommandReturnObject ()
{
    // m_opaque_ap will automatically delete any pointer it owns
}

bool
SBCommandReturnObject::IsValid() const
{
    return m_opaque_ap.get() != NULL;
}


const char *
SBCommandReturnObject::GetOutput ()
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (m_opaque_ap.get())
    {
        if (log)
            log->Printf ("SBCommandReturnObject(%p)::GetOutput () => \"%s\"", m_opaque_ap.get(), 
                         m_opaque_ap->GetOutputData());

        return m_opaque_ap->GetOutputData();
    }

    if (log)
        log->Printf ("SBCommandReturnObject(%p)::GetOutput () => NULL", m_opaque_ap.get());

    return NULL;
}

const char *
SBCommandReturnObject::GetError ()
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (m_opaque_ap.get())
    {
        if (log)
            log->Printf ("SBCommandReturnObject(%p)::GetError () => \"%s\"", m_opaque_ap.get(),
                         m_opaque_ap->GetErrorData());

        return m_opaque_ap->GetErrorData();
    }
    
    if (log)
        log->Printf ("SBCommandReturnObject(%p)::GetError () => NULL", m_opaque_ap.get());

    return NULL;
}

size_t
SBCommandReturnObject::GetOutputSize ()
{
    if (m_opaque_ap.get())
        return strlen (m_opaque_ap->GetOutputData());
    return 0;
}

size_t
SBCommandReturnObject::GetErrorSize ()
{
    if (m_opaque_ap.get())
        return strlen(m_opaque_ap->GetErrorData());
    return 0;
}

size_t
SBCommandReturnObject::PutOutput (FILE *fh)
{
    if (fh)
    {
        size_t num_bytes = GetOutputSize ();
        if (num_bytes)
            return ::fprintf (fh, "%s", GetOutput());
    }
    return 0;
}

size_t
SBCommandReturnObject::PutError (FILE *fh)
{
    if (fh)
    {
        size_t num_bytes = GetErrorSize ();
        if (num_bytes)
            return ::fprintf (fh, "%s", GetError());
    }
    return 0;
}

void
SBCommandReturnObject::Clear()
{
    if (m_opaque_ap.get())
        m_opaque_ap->Clear();
}

lldb::ReturnStatus
SBCommandReturnObject::GetStatus()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetStatus();
    return lldb::eReturnStatusInvalid;
}

void
SBCommandReturnObject::SetStatus(lldb::ReturnStatus status)
{
    if (m_opaque_ap.get())
         m_opaque_ap->SetStatus(status);
}

bool
SBCommandReturnObject::Succeeded ()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->Succeeded();
    return false;
}

bool
SBCommandReturnObject::HasResult ()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->HasResult();
    return false;
}

void
SBCommandReturnObject::AppendMessage (const char *message)
{
    if (m_opaque_ap.get())
        m_opaque_ap->AppendMessage (message);
}

void
SBCommandReturnObject::AppendWarning (const char *message)
{
    if (m_opaque_ap.get())
        m_opaque_ap->AppendWarning (message);
}

CommandReturnObject *
SBCommandReturnObject::operator ->() const
{
    return m_opaque_ap.get();
}

CommandReturnObject *
SBCommandReturnObject::get() const
{
    return m_opaque_ap.get();
}

CommandReturnObject &
SBCommandReturnObject::operator *() const
{
    assert(m_opaque_ap.get());
    return *(m_opaque_ap.get());
}


CommandReturnObject &
SBCommandReturnObject::ref() const
{
    assert(m_opaque_ap.get());
    return *(m_opaque_ap.get());
}


void
SBCommandReturnObject::SetLLDBObjectPtr (CommandReturnObject *ptr)
{
    if (m_opaque_ap.get())
        m_opaque_ap.reset (ptr);
}

bool
SBCommandReturnObject::GetDescription (SBStream &description)
{
    Stream &strm = description.ref();

    if (m_opaque_ap.get())
    {
        description.Printf ("Status:  ");
        lldb::ReturnStatus status = m_opaque_ap->GetStatus();
        if (status == lldb::eReturnStatusStarted)
            strm.PutCString ("Started");
        else if (status == lldb::eReturnStatusInvalid)
            strm.PutCString ("Invalid");
        else if (m_opaque_ap->Succeeded())
            strm.PutCString ("Success");
        else
            strm.PutCString ("Fail");

        if (GetOutputSize() > 0)
            strm.Printf ("\nOutput Message:\n%s", GetOutput());

        if (GetErrorSize() > 0)
            strm.Printf ("\nError Message:\n%s", GetError());
    }
    else
        strm.PutCString ("No value");

    return true;
}

void
SBCommandReturnObject::SetImmediateOutputFile (FILE *fh)
{
    if (m_opaque_ap.get())
        m_opaque_ap->SetImmediateOutputFile (fh);
}

void
SBCommandReturnObject::SetImmediateErrorFile (FILE *fh)
{
    if (m_opaque_ap.get())
        m_opaque_ap->SetImmediateErrorFile (fh);
}

void
SBCommandReturnObject::PutCString(const char* string, int len)
{
    if (m_opaque_ap.get())
    {
        if (len == 0 || string == NULL || *string == 0)
        {
            return;
        }
        else if (len > 0)
        {
            std::string buffer(string, len);
            m_opaque_ap->AppendMessage(buffer.c_str());
        }
        else
            m_opaque_ap->AppendMessage(string);
    }
}

const char *
SBCommandReturnObject::GetOutput (bool only_if_no_immediate)
{
    if (!m_opaque_ap.get())
        return NULL;
    if (only_if_no_immediate == false || m_opaque_ap->GetImmediateOutputStream().get() == NULL)
        return GetOutput();
    return NULL;
}

const char *
SBCommandReturnObject::GetError (bool only_if_no_immediate)
{
    if (!m_opaque_ap.get())
        return NULL;
    if (only_if_no_immediate == false || m_opaque_ap->GetImmediateErrorStream().get() == NULL)
        return GetError();
    return NULL;
}

size_t
SBCommandReturnObject::Printf(const char* format, ...)
{
    if (m_opaque_ap.get())
    {
        va_list args;
        va_start (args, format);
        size_t result = m_opaque_ap->GetOutputStream().PrintfVarArg(format, args);
        va_end (args);
        return result;
    }
    return 0;
}

