//===-- Error.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdarg>
#include <cstdlib>
#include <cstring>

#if defined (__arm__)
#include <SpringBoardServices/SpringBoardServer.h>
#endif

using namespace lldb;
using namespace lldb_private;

Error::Error ():
    m_code (0),
    m_type (eErrorTypeInvalid),
    m_string ()
{
}

//----------------------------------------------------------------------
// Default constructor
//----------------------------------------------------------------------
Error::Error(ValueType err, ErrorType type) :
    m_code (err),
    m_type (type),
    m_string ()
{
}

Error::Error (const Error &rhs) :
    m_code (rhs.m_code),
    m_type (rhs.m_type),
    m_string (rhs.m_string)
{
}

Error::Error (const char* err_str):
    m_code (0),
    m_type (eErrorTypeInvalid),
    m_string ()
{
    SetErrorString(err_str);
}

//----------------------------------------------------------------------
// Assignment operator
//----------------------------------------------------------------------
const Error&
Error::operator = (const Error& rhs)
{
    if (this != &rhs)
    {
        m_code = rhs.m_code;
        m_type = rhs.m_type;
        m_string = rhs.m_string;
    }
    return *this;
}


//----------------------------------------------------------------------
// Assignment operator
//----------------------------------------------------------------------
const Error&
Error::operator = (uint32_t err)
{
    m_code = err;
    m_type = eErrorTypeMachKernel;
    m_string.clear();
    return *this;
}

Error::~Error()
{
}

//----------------------------------------------------------------------
// Get the error value as a NULL C string. The error string will be
// fetched and cached on demand. The cached error string value will
// remain until the error value is changed or cleared.
//----------------------------------------------------------------------
const char *
Error::AsCString(const char *default_error_str) const
{
    if (Success())
        return NULL;

    if (m_string.empty())
    {
        const char *s = NULL;
        switch (m_type)
        {
        case eErrorTypeMachKernel:
#if defined (__APPLE__)
            s = ::mach_error_string (m_code);
#endif
            break;

        case eErrorTypePOSIX:
            s = ::strerror (m_code);
            break;

        default:
            break;
        }
        if (s)
            m_string.assign(s);
    }
    if (m_string.empty())
    {
        if (default_error_str)
            m_string.assign(default_error_str);
        else
            return NULL;    // User wanted a NULL string back...
    }
    return m_string.c_str();
}


//----------------------------------------------------------------------
// Clear the error and any cached error string that it might contain.
//----------------------------------------------------------------------
void
Error::Clear ()
{
    m_code = 0;
    m_type = eErrorTypeGeneric;
    m_string.clear();
}

//----------------------------------------------------------------------
// Access the error value.
//----------------------------------------------------------------------
Error::ValueType
Error::GetError () const
{
    return m_code;
}

//----------------------------------------------------------------------
// Access the error type.
//----------------------------------------------------------------------
ErrorType
Error::GetType () const
{
    return m_type;
}

//----------------------------------------------------------------------
// Retuns true if this object contains an value that describes an
// error or otherwise non-success result.
//----------------------------------------------------------------------
bool
Error::Fail () const
{
    return m_code != 0;
}

//----------------------------------------------------------------------
// Log the error given a string with format. If the this object
// contains an error code, update the error string to contain the
// "error: " followed by the formatted string, followed by the error
// value and any string that describes the current error. This
// allows more context to be given to an error string that remains
// cached in this object. Logging always occurs even when the error
// code contains a non-error value.
//----------------------------------------------------------------------
void
Error::PutToLog (Log *log, const char *format, ...)
{
    char *arg_msg = NULL;
    va_list args;
    va_start (args, format);
    ::vasprintf (&arg_msg, format, args);
    va_end (args);

    if (arg_msg != NULL)
    {
        if (Fail())
        {
            const char *err_str = AsCString();
            if (err_str == NULL)
                err_str = "???";

            SetErrorStringWithFormat("error: %s err = %s (0x%8.8x)", arg_msg, err_str, m_code);
            if (log)
                log->Error("%s", m_string.c_str());
        }
        else
        {
            if (log)
                log->Printf("%s err = 0x%8.8x", arg_msg, m_code);
        }
        ::free (arg_msg);
    }
}

//----------------------------------------------------------------------
// Log the error given a string with format. If the this object
// contains an error code, update the error string to contain the
// "error: " followed by the formatted string, followed by the error
// value and any string that describes the current error. This
// allows more context to be given to an error string that remains
// cached in this object. Logging only occurs even when the error
// code contains a error value.
//----------------------------------------------------------------------
void
Error::LogIfError (Log *log, const char *format, ...)
{
    if (Fail())
    {
        char *arg_msg = NULL;
        va_list args;
        va_start (args, format);
        ::vasprintf (&arg_msg, format, args);
        va_end (args);

        if (arg_msg != NULL)
        {
            const char *err_str = AsCString();
            if (err_str == NULL)
                err_str = "???";

            SetErrorStringWithFormat("error: %s err = %s (0x%8.8x)", arg_msg, err_str, m_code);
            if (log)
                log->Error("%s", m_string.c_str());

            ::free (arg_msg);
        }
    }
}

//----------------------------------------------------------------------
// Set accesssor for the error value to "err" and the type to
// "eErrorTypeMachKernel"
//----------------------------------------------------------------------
void
Error::SetMachError (uint32_t err)
{
    m_code = err;
    m_type = eErrorTypeMachKernel;
    m_string.clear();
}

//----------------------------------------------------------------------
// Set accesssor for the error value and type.
//----------------------------------------------------------------------
void
Error::SetError (ValueType err, ErrorType type)
{
    m_code = err;
    m_type = type;
    m_string.clear();
}

//----------------------------------------------------------------------
// Update the error value to be "errno" and update the type to
// be "POSIX".
//----------------------------------------------------------------------
void
Error::SetErrorToErrno()
{
    m_code = errno;
    m_type = eErrorTypePOSIX;
    m_string.clear();
}

//----------------------------------------------------------------------
// Update the error value to be LLDB_GENERIC_ERROR and update the type
// to be "Generic".
//----------------------------------------------------------------------
void
Error::SetErrorToGenericError ()
{
    m_code = LLDB_GENERIC_ERROR;
    m_type = eErrorTypeGeneric;
    m_string.clear();
}

//----------------------------------------------------------------------
// Set accessor for the error string value for a specific error.
// This allows any string to be supplied as an error explanation.
// The error string value will remain until the error value is
// cleared or a new error value/type is assigned.
//----------------------------------------------------------------------
void
Error::SetErrorString (const char *err_str)
{
    if (err_str && err_str[0])
    {
        // If we have an error string, we should always at least have
        // an error set to a generic value.
        if (Success())
            SetErrorToGenericError();
        m_string = err_str;
    }
    else
        m_string.clear();
}

//------------------------------------------------------------------
/// Set the current error string to a formatted error string.
///
/// @param format
///     A printf style format string
//------------------------------------------------------------------
int
Error::SetErrorStringWithFormat (const char *format, ...)
{
    if (format && format[0])
    {
        va_list args;
        va_start (args, format);
        int length = SetErrorStringWithVarArg (format, args);
        va_end (args);
        return length;
    }
    else
    {
        m_string.clear();
    }
    return 0;
}

int
Error::SetErrorStringWithVarArg (const char *format, va_list args)
{
    if (format && format[0])
    {
        // If we have an error string, we should always at least have
        // an error set to a generic value.
        if (Success())
            SetErrorToGenericError();

        // Try and fit our error into a 1024 byte buffer first...
        llvm::SmallVector<char, 1024> buf;
        buf.resize(1024);
        // Copy in case our first call to vsnprintf doesn't fit into our
        // allocated buffer above
        va_list copy_args;
        va_copy (copy_args, args);
        size_t length = ::vsnprintf (buf.data(), buf.size(), format, args);
        if (length >= buf.size())
        {
            // The error formatted string didn't fit into our buffer, resize it
            // to the exact needed size, and retry
            buf.resize(length + 1);
            length = ::vsnprintf (buf.data(), buf.size(), format, copy_args);
            va_end (copy_args);
            assert (length < buf.size());
        }
        m_string.assign(buf.data(), length);
        va_end (args);
        return length;
    }
    else
    {
        m_string.clear();
    }
    return 0;
}


//----------------------------------------------------------------------
// Returns true if the error code in this object is considered a
// successful return value.
//----------------------------------------------------------------------
bool
Error::Success() const
{
    return m_code == 0;
}
