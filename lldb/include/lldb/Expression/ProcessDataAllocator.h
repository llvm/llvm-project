//===-- ProcessDataAllocator.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessDataAllocator_h_
#define liblldb_ProcessDataAllocator_h_

#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "lldb/Expression/IRForTarget.h"
#include "lldb/Target/Process.h"

namespace lldb_private 
{

class ProcessDataAllocator : public IRForTarget::StaticDataAllocator {
public:
    ProcessDataAllocator(Process &process) : 
        IRForTarget::StaticDataAllocator(),
        m_process(process),
        m_stream_string(StreamString::eBinary, process.GetAddressByteSize(), process.GetByteOrder()),
        m_allocation(0)
    {
    }
    
    ~ProcessDataAllocator()
    {
        if (m_allocation)
            m_process.DeallocateMemory(m_allocation);
    }
    
    lldb_private::StreamString &GetStream()
    {
        return m_stream_string;
    }
    
    lldb::addr_t Allocate()
    {
        Error err;
        
        if (m_allocation)
            m_process.DeallocateMemory(m_allocation);
        
        m_allocation = m_process.AllocateMemory(m_stream_string.GetSize(), lldb::ePermissionsReadable | lldb::ePermissionsWritable, err);
        
        if (!err.Success())
            return 0;
        
        if (m_allocation)
            m_process.WriteMemory(m_allocation, m_stream_string.GetData(), m_stream_string.GetSize(), err);
        
        if (!err.Success())
            return 0;
        
        return m_allocation;
    }
    
    void Dump(lldb_private::Stream &stream);
private:
    Process        &m_process;
    StreamString    m_stream_string;
    lldb::addr_t    m_allocation;
};
    
} // namespace lldb_private

#endif
