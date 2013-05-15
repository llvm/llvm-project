//===-- SBInputReader.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBInputReader_h_
#define LLDB_SBInputReader_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBInputReader
{
public:

    typedef size_t (*Callback) (void *baton, 
                                SBInputReader *reader, 
                                InputReaderAction notification,
                                const char *bytes, 
                                size_t bytes_len);

    SBInputReader ();

    SBInputReader (const lldb::InputReaderSP &reader_sp);

    SBInputReader (const lldb::SBInputReader &rhs);

    ~SBInputReader ();


    SBError
    Initialize (SBDebugger &debugger,
                Callback callback,
                void *callback_baton,
                lldb::InputReaderGranularity granularity,
                const char *end_token,
                const char *prompt,
                bool echo);
    
    bool
    IsValid () const;

    const lldb::SBInputReader &
    operator = (const lldb::SBInputReader &rhs);

    bool
    IsActive () const;

    bool
    IsDone () const;

    void
    SetIsDone (bool value);

    InputReaderGranularity
    GetGranularity ();

protected:
    friend class SBDebugger;

    lldb_private::InputReader *
    operator->() const;

    lldb::InputReaderSP &
    operator *();

    const lldb::InputReaderSP &
    operator *() const;

    lldb_private::InputReader *
    get() const;

    lldb_private::InputReader &
    ref() const;

private:

    static size_t
    PrivateCallback (void *baton, 
                     lldb_private::InputReader &reader, 
                     lldb::InputReaderAction notification,
                     const char *bytes, 
                     size_t bytes_len);

    lldb::InputReaderSP m_opaque_sp;
    Callback m_callback_function;
    void *m_callback_baton;
};

} // namespace lldb

#endif // LLDB_SBInputReader_h_
