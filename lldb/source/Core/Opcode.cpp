//===-- Opcode.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Opcode.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/ADT/Triple.h"
// Project includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Stream.h"
#include "lldb/Host/Endian.h"


using namespace lldb;
using namespace lldb_private;


int
Opcode::Dump (Stream *s, uint32_t min_byte_width)
{
    int bytes_written = 0;
    switch (m_type)
    {
    case Opcode::eTypeInvalid:  
        bytes_written = s->PutCString ("<invalid>"); 
        break;
    case Opcode::eType8:        
        bytes_written = s->Printf ("0x%2.2x", m_data.inst8); 
        break;
    case Opcode::eType16:
        bytes_written = s->Printf ("0x%4.4x", m_data.inst16); 
        break;
    case Opcode::eType16_2:
    case Opcode::eType32:
        bytes_written = s->Printf ("0x%8.8x", m_data.inst32); 
        break;

    case Opcode::eType64:
        bytes_written = s->Printf ("0x%16.16" PRIx64, m_data.inst64);
        break;

    case Opcode::eTypeBytes:
        {
            for (uint32_t i=0; i<m_data.inst.length; ++i)
            {
                if (i > 0)
                    bytes_written += s->PutChar (' ');
                bytes_written += s->Printf ("%2.2x", m_data.inst.bytes[i]); 
            }
        }
        break;
    }
    
    // Add spaces to make sure bytes dispay comes out even in case opcodes
    // aren't all the same size
    if (bytes_written < min_byte_width)
        bytes_written = s->Printf ("%*s", min_byte_width - bytes_written, "");
    return bytes_written;
}

lldb::ByteOrder
Opcode::GetDataByteOrder () const
{
    switch (m_type)
    {
        case Opcode::eTypeInvalid: break;
        case Opcode::eType8:
        case Opcode::eType16:
        case Opcode::eType16_2:
        case Opcode::eType32:
        case Opcode::eType64:    return lldb::endian::InlHostByteOrder();
        case Opcode::eTypeBytes:
            break;
    }
    return eByteOrderInvalid;
}

uint32_t
Opcode::GetData (DataExtractor &data) const
{
    uint32_t byte_size = GetByteSize ();
    
    DataBufferSP buffer_sp;
    if (byte_size > 0)
    {
        switch (m_type)
        {
            case Opcode::eTypeInvalid:
                break;
                
            case Opcode::eType8:    buffer_sp.reset (new DataBufferHeap (&m_data.inst8,  byte_size)); break;
            case Opcode::eType16:   buffer_sp.reset (new DataBufferHeap (&m_data.inst16, byte_size)); break;
            case Opcode::eType16_2:
                {
                    // 32 bit thumb instruction, we need to sizzle this a bit
                    uint8_t buf[4];
                    buf[0] = m_data.inst.bytes[2];
                    buf[1] = m_data.inst.bytes[3];
                    buf[2] = m_data.inst.bytes[0];
                    buf[3] = m_data.inst.bytes[1];
                    buffer_sp.reset (new DataBufferHeap (buf, byte_size));
                }
                break;
            case Opcode::eType32:
                buffer_sp.reset (new DataBufferHeap (&m_data.inst32, byte_size));
                break;
            case Opcode::eType64:   buffer_sp.reset (new DataBufferHeap (&m_data.inst64, byte_size)); break;
            case Opcode::eTypeBytes:buffer_sp.reset (new DataBufferHeap (GetOpcodeBytes(), byte_size)); break;
                break;
        }
    }
    
    if (buffer_sp)
    {
        data.SetByteOrder(GetDataByteOrder());
        data.SetData (buffer_sp);
        return byte_size;
    }
    data.Clear();
    return 0;
}



