//===-- InputReaderEZ.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/InputReaderEZ.h"

using namespace lldb;
using namespace lldb_private;

size_t
InputReaderEZ::Callback_Impl(void *baton, 
                             InputReader &reader, 
                             lldb::InputReaderAction notification,
                             const char *bytes, 
                             size_t bytes_len)

{
    HandlerData hand_data(reader,
                          bytes,
                          bytes_len,
                          baton);
    
    switch (notification)
    {
        case eInputReaderActivate:
            reader.ActivateHandler(hand_data);
            break;
        case eInputReaderDeactivate:
            reader.DeactivateHandler(hand_data);
            break;
        case eInputReaderReactivate:
            reader.ReactivateHandler(hand_data);
            break;
        case eInputReaderAsynchronousOutputWritten:
            reader.AsynchronousOutputWrittenHandler(hand_data);
            break;
        case eInputReaderGotToken:
        {
            if (reader.GetSaveUserInput())
                reader.GetUserInput().AppendString(bytes, bytes_len);            
            reader.GotTokenHandler(hand_data);
        }
            break;
        case eInputReaderInterrupt:
            reader.InterruptHandler(hand_data);
            break;
        case eInputReaderEndOfFile:
            reader.EOFHandler(hand_data);
            break;
        case eInputReaderDone:
            reader.DoneHandler(hand_data);
            break;
    }
    return bytes_len;
}

Error
InputReaderEZ::Initialize(void* baton,
                          lldb::InputReaderGranularity token_size,
                          const char* end_token,
                          const char *prompt,
                          bool echo)
{
    return InputReader::Initialize(Callback_Impl,
                                   baton,
                                   token_size,
                                   end_token,
                                   prompt,
                                   echo);
}

Error
InputReaderEZ::Initialize(InitializationParameters& params)
{
    Error ret =  Initialize(params.m_baton,
                            params.m_token_size,
                            params.m_end_token,
                            params.m_prompt,
                            params.m_echo);
    m_save_user_input = params.m_save_user_input;
    return ret;
}

InputReaderEZ::~InputReaderEZ ()
{
}
