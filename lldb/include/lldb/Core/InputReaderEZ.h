//===-- InputReaderEZ.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_InputReaderEZ_h_
#define liblldb_InputReaderEZ_h_

#include "lldb/Core/InputReader.h"

namespace lldb_private {

class InputReaderEZ : public InputReader
{

private:
    
    static size_t Callback_Impl(void *baton,
                                InputReader &reader,
                                lldb::InputReaderAction notification,
                                const char *bytes,
                                size_t bytes_len);
public:
    
    InputReaderEZ (Debugger &debugger) :
    InputReader(debugger)
    {}

    virtual
    ~InputReaderEZ ();

    using InputReader::Initialize;
    virtual Error
    Initialize(void* baton,
               lldb::InputReaderGranularity token_size = lldb::eInputReaderGranularityLine,
               const char* end_token = "DONE",
               const char *prompt = "> ",
               bool echo = true);
        
    virtual Error
    Initialize(InitializationParameters& params);
    
    virtual void
    ActivateHandler(HandlerData&) {}
    
    virtual void
    DeactivateHandler(HandlerData&) {}
    
    virtual void
    ReactivateHandler(HandlerData&) {}
    
    virtual void
    AsynchronousOutputWrittenHandler(HandlerData&) {}
    
    virtual void
    GotTokenHandler(HandlerData&) {}
    
    virtual void
    InterruptHandler(HandlerData&) {}
    
    virtual void
    EOFHandler(HandlerData&) {}
    
    virtual void
    DoneHandler(HandlerData&) {}
    
    virtual bool
    GetSaveUserInput()
    {
        return m_save_user_input;
    }
    
protected:
    friend class Debugger;

private:
    DISALLOW_COPY_AND_ASSIGN (InputReaderEZ);

};

} // namespace lldb_private

#endif // #ifndef liblldb_InputReaderEZ_h_
