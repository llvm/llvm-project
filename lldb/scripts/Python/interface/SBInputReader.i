//===-- SWIG Interface for SBInputREader ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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

    bool
    IsActive () const;

    bool
    IsDone () const;

    void
    SetIsDone (bool value);

    InputReaderGranularity
    GetGranularity ();
};

} // namespace lldb
