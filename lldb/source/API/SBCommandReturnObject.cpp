//===-- SBCommandReturnObject.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBStream.h"

#include "lldb/Core/Error.h"
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
    if (rhs.m_opaque_ap)
        m_opaque_ap.reset (new CommandReturnObject (*rhs.m_opaque_ap));
}

SBCommandReturnObject::SBCommandReturnObject (CommandReturnObject *ptr) :
    m_opaque_ap (ptr)
{
}

SBCommandReturnObject::~SBCommandReturnObject() = default;

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
        if (rhs.m_opaque_ap)
            m_opaque_ap.reset (new CommandReturnObject (*rhs.m_opaque_ap));
        else
            m_opaque_ap.reset();
    }
    return *this;
}

bool
SBCommandReturnObject::IsValid() const
{
    return m_opaque_ap.get() != nullptr;
}

const char *
SBCommandReturnObject::GetOutput ()
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (m_opaque_ap)
    {
        if (log)
            log->Printf ("SBCommandReturnObject(%p)::GetOutput () => \"%s\"",
                         static_cast<void*>(m_opaque_ap.get()),
                         m_opaque_ap->GetOutputData());

        return m_opaque_ap->GetOutputData();
    }

    if (log)
        log->Printf ("SBCommandReturnObject(%p)::GetOutput () => nullptr",
                     static_cast<void*>(m_opaque_ap.get()));

    return nullptr;
}

const char *
SBCommandReturnObject::GetError ()
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (m_opaque_ap)
    {
        if (log)
            log->Printf ("SBCommandReturnObject(%p)::GetError () => \"%s\"",
                         static_cast<void*>(m_opaque_ap.get()),
                         m_opaque_ap->GetErrorData());

        return m_opaque_ap->GetErrorData();
    }

    if (log)
        log->Printf ("SBCommandReturnObject(%p)::GetError () => nullptr",
                     static_cast<void*>(m_opaque_ap.get()));

    return nullptr;
}

size_t
SBCommandReturnObject::GetOutputSize()
{
    return (m_opaque_ap ? strlen(m_opaque_ap->GetOutputData()) : 0);
}

size_t
SBCommandReturnObject::GetErrorSize()
{
    return (m_opaque_ap ? strlen(m_opaque_ap->GetErrorData()) : 0);
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
    if (m_opaque_ap)
        m_opaque_ap->Clear();
}

lldb::ReturnStatus
SBCommandReturnObject::GetStatus()
{
    return (m_opaque_ap ? m_opaque_ap->GetStatus() : lldb::eReturnStatusInvalid);
}

void
SBCommandReturnObject::SetStatus(lldb::ReturnStatus status)
{
    if (m_opaque_ap)
         m_opaque_ap->SetStatus(status);
}

bool
SBCommandReturnObject::Succeeded()
{
    return (m_opaque_ap ? m_opaque_ap->Succeeded() : false);
}

bool
SBCommandReturnObject::HasResult()
{
    return (m_opaque_ap ? m_opaque_ap->HasResult() : false);
}

void
SBCommandReturnObject::AppendMessage (const char *message)
{
    if (m_opaque_ap)
        m_opaque_ap->AppendMessage (message);
}

void
SBCommandReturnObject::AppendWarning (const char *message)
{
    if (m_opaque_ap)
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
    if (m_opaque_ap)
        m_opaque_ap.reset (ptr);
}

bool
SBCommandReturnObject::GetDescription (SBStream &description)
{
    Stream &strm = description.ref();

    if (m_opaque_ap)
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
SBCommandReturnObject::SetImmediateOutputFile(FILE *fh)
{
    SetImmediateOutputFile(fh, false);
}

void
SBCommandReturnObject::SetImmediateErrorFile(FILE *fh)
{
    SetImmediateErrorFile(fh, false);
}

void
SBCommandReturnObject::SetImmediateOutputFile(FILE *fh, bool transfer_ownership)
{
    if (m_opaque_ap)
        m_opaque_ap->SetImmediateOutputFile(fh, transfer_ownership);
}

void
SBCommandReturnObject::SetImmediateErrorFile(FILE *fh, bool transfer_ownership)
{
    if (m_opaque_ap)
        m_opaque_ap->SetImmediateErrorFile(fh, transfer_ownership);
}

void
SBCommandReturnObject::PutCString(const char* string, int len)
{
    if (m_opaque_ap)
    {
        if (len == 0 || string == nullptr || *string == 0)
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
    if (!m_opaque_ap)
        return nullptr;
    if (!only_if_no_immediate || m_opaque_ap->GetImmediateOutputStream().get() == nullptr)
        return GetOutput();
    return nullptr;
}

const char *
SBCommandReturnObject::GetError (bool only_if_no_immediate)
{
    if (!m_opaque_ap)
        return nullptr;
    if (!only_if_no_immediate || m_opaque_ap->GetImmediateErrorStream().get() == nullptr)
        return GetError();
    return nullptr;
}

size_t
SBCommandReturnObject::Printf(const char* format, ...)
{
    if (m_opaque_ap)
    {
        va_list args;
        va_start (args, format);
        size_t result = m_opaque_ap->GetOutputStream().PrintfVarArg(format, args);
        va_end (args);
        return result;
    }
    return 0;
}

void
SBCommandReturnObject::SetError (lldb::SBError &error, const char *fallback_error_cstr)
{
    if (m_opaque_ap)
    {
        if (error.IsValid())
            m_opaque_ap->SetError(error.ref(), fallback_error_cstr);
        else if (fallback_error_cstr)
            m_opaque_ap->SetError(Error(), fallback_error_cstr);
    }
}

void
SBCommandReturnObject::SetError (const char *error_cstr)
{
    if (m_opaque_ap && error_cstr)
        m_opaque_ap->SetError(error_cstr);
}
