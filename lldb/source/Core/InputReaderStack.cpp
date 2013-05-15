//===-- InputReaderStack.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/InputReaderStack.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes


using namespace lldb;
using namespace lldb_private;

InputReaderStack::InputReaderStack () :
    m_input_readers (),
    m_input_readers_mutex (Mutex::eMutexTypeRecursive)
{
}

InputReaderStack::~InputReaderStack ()
{
}

size_t
InputReaderStack::GetSize () const
{
    Mutex::Locker locker (m_input_readers_mutex);
    return m_input_readers.size();
}
    
void
InputReaderStack::Push (const lldb::InputReaderSP& reader_sp)
{
    if (reader_sp)
    {
        Mutex::Locker locker (m_input_readers_mutex);
        m_input_readers.push (reader_sp);
    }
}
    
bool
InputReaderStack::IsEmpty () const
{
    Mutex::Locker locker (m_input_readers_mutex);
    return m_input_readers.empty();
}
    
InputReaderSP
InputReaderStack::Top ()
{
    InputReaderSP input_reader_sp;
    {
        Mutex::Locker locker (m_input_readers_mutex);
        if (!m_input_readers.empty())
            input_reader_sp = m_input_readers.top();
    }
        
    return input_reader_sp;
}
    
void
InputReaderStack::Pop ()
{
    Mutex::Locker locker (m_input_readers_mutex);
    if (!m_input_readers.empty())
        m_input_readers.pop();
}
    
Mutex &
InputReaderStack::GetStackMutex ()
{
    return m_input_readers_mutex;
}
