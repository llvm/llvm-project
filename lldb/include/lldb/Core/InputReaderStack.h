//===-- InputReaderStack.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_InputReaderStack_h_
#define liblldb_InputReaderStack_h_

#include <stack>

#include "lldb/lldb-private.h"
#include "lldb/Host/Mutex.h"

namespace lldb_private {

class InputReaderStack
{
public:

    InputReaderStack ();
    
    ~InputReaderStack ();

    size_t
    GetSize () const;
    
    void
    Push (const lldb::InputReaderSP& reader_sp);
    
    bool
    IsEmpty () const;
    
    lldb::InputReaderSP
    Top ();
    
    void
    Pop ();
    
    Mutex &
    GetStackMutex ();
    
protected:

    std::stack<lldb::InputReaderSP> m_input_readers;
    mutable Mutex m_input_readers_mutex;
    
private:

    DISALLOW_COPY_AND_ASSIGN (InputReaderStack);
};

} // namespace lldb_private

#endif // liblldb_InputReaderStack_h_
