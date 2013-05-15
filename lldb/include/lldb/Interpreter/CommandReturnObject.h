//===-- CommandReturnObject.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandReturnObject_h_
#define liblldb_CommandReturnObject_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/STLUtils.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/StreamTee.h"

namespace lldb_private {


class CommandReturnObject
{
public:

    CommandReturnObject ();
    
    ~CommandReturnObject ();

    const char *
    GetOutputData ()
    {
        lldb::StreamSP stream_sp (m_out_stream.GetStreamAtIndex (eStreamStringIndex));
        if (stream_sp)
            return static_cast<StreamString *>(stream_sp.get())->GetData();
        return "";
    }

    const char *
    GetErrorData ()
    {
        lldb::StreamSP stream_sp (m_err_stream.GetStreamAtIndex (eStreamStringIndex));
        if (stream_sp)
            return static_cast<StreamString *>(stream_sp.get())->GetData();
        else
            return "";
    }

    Stream &
    GetOutputStream ()
    {
        // Make sure we at least have our normal string stream output stream
        lldb::StreamSP stream_sp (m_out_stream.GetStreamAtIndex (eStreamStringIndex));
        if (!stream_sp)
        {
            stream_sp.reset (new StreamString());
            m_out_stream.SetStreamAtIndex (eStreamStringIndex, stream_sp);
        }   
        return m_out_stream;
    }

    Stream &
    GetErrorStream ()
    {
        // Make sure we at least have our normal string stream output stream
        lldb::StreamSP stream_sp (m_err_stream.GetStreamAtIndex (eStreamStringIndex));
        if (!stream_sp)
        {
            stream_sp.reset (new StreamString());
            m_err_stream.SetStreamAtIndex (eStreamStringIndex, stream_sp);
        }   
        return m_err_stream;
    }

    void
    SetImmediateOutputFile (FILE *fh, bool transfer_fh_ownership = false)
    {
        lldb::StreamSP stream_sp (new StreamFile (fh, transfer_fh_ownership));
        m_out_stream.SetStreamAtIndex (eImmediateStreamIndex, stream_sp);
    }
    
    void
    SetImmediateErrorFile (FILE *fh, bool transfer_fh_ownership = false)
    {
        lldb::StreamSP stream_sp (new StreamFile (fh, transfer_fh_ownership));
        m_err_stream.SetStreamAtIndex (eImmediateStreamIndex, stream_sp);
    }
    
    void
    SetImmediateOutputStream (const lldb::StreamSP &stream_sp)
    {
        m_out_stream.SetStreamAtIndex (eImmediateStreamIndex, stream_sp);
    }
    
    void
    SetImmediateErrorStream (const lldb::StreamSP &stream_sp)
    {
        m_err_stream.SetStreamAtIndex (eImmediateStreamIndex, stream_sp);
    }
    
    lldb::StreamSP
    GetImmediateOutputStream ()
    {
        return m_out_stream.GetStreamAtIndex (eImmediateStreamIndex);
    }
    
    lldb::StreamSP
    GetImmediateErrorStream ()
    {
        return m_err_stream.GetStreamAtIndex (eImmediateStreamIndex);
    }
    
    void
    Clear();

    void
    AppendMessage (const char *in_string);

    void
    AppendMessageWithFormat (const char *format, ...)  __attribute__ ((format (printf, 2, 3)));

    void
    AppendRawWarning (const char *in_string);

    void
    AppendWarning (const char *in_string);

    void
    AppendWarningWithFormat (const char *format, ...)  __attribute__ ((format (printf, 2, 3)));

    void
    AppendError (const char *in_string);

    void
    AppendRawError (const char *in_string);

    void
    AppendErrorWithFormat (const char *format, ...)  __attribute__ ((format (printf, 2, 3)));

    void
    SetError (const Error &error, 
              const char *fallback_error_cstr);

    lldb::ReturnStatus
    GetStatus();

    void
    SetStatus (lldb::ReturnStatus status);

    bool
    Succeeded ();

    bool
    HasResult ();

    bool GetDidChangeProcessState ();

    void SetDidChangeProcessState (bool b);

private:
    enum 
    {
        eStreamStringIndex = 0,
        eImmediateStreamIndex = 1
    };
    
    StreamTee    m_out_stream;
    StreamTee    m_err_stream;
    
    lldb::ReturnStatus m_status;
    bool m_did_change_process_state;
};

} // namespace lldb_private

#endif  // liblldb_CommandReturnObject_h_
