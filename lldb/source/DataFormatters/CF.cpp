//===-- CF.cpp ----------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/CXXFormatterFunctions.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

bool
lldb_private::formatters::CFAbsoluteTimeSummaryProvider (ValueObject& valobj, Stream& stream)
{
    time_t epoch = GetOSXEpoch();
    epoch = epoch + (time_t)valobj.GetValueAsUnsigned(0);
    tm *tm_date = localtime(&epoch);
    if (!tm_date)
        return false;
    std::string buffer(1024,0);
    if (strftime (&buffer[0], 1023, "%Z", tm_date) == 0)
        return false;
    stream.Printf("%04d-%02d-%02d %02d:%02d:%02d %s", tm_date->tm_year+1900, tm_date->tm_mon+1, tm_date->tm_mday, tm_date->tm_hour, tm_date->tm_min, tm_date->tm_sec, buffer.c_str());
    return true;
}

bool
lldb_private::formatters::CFBagSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    uint32_t count = 0;
    
    bool is_type_ok = false; // check to see if this is a CFBag we know about
    if (descriptor->IsCFType())
    {
        ConstString type_name(valobj.GetTypeName());
        if (type_name == ConstString("__CFBag") || type_name == ConstString("const struct __CFBag"))
        {
            if (valobj.IsPointerType())
                is_type_ok = true;
        }
    }
    
    if (is_type_ok == false)
    {
        StackFrameSP frame_sp(valobj.GetFrameSP());
        if (!frame_sp)
            return false;
        ValueObjectSP count_sp;
        StreamString expr;
        expr.Printf("(int)CFBagGetCount((void*)0x%" PRIx64 ")",valobj.GetPointerValue());
        if (process_sp->GetTarget().EvaluateExpression(expr.GetData(), frame_sp.get(), count_sp) != eExecutionCompleted)
            return false;
        if (!count_sp)
            return false;
        count = count_sp->GetValueAsUnsigned(0);
    }
    else
    {
        uint32_t offset = 2*ptr_size+4 + valobj_addr;
        Error error;
        count = process_sp->ReadUnsignedIntegerFromMemory(offset, 4, 0, error);
        if (error.Fail())
            return false;
    }
    stream.Printf("@\"%u value%s\"",
                  count,(count == 1 ? "" : "s"));
    return true;
}

bool
lldb_private::formatters::CFBitVectorSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    uint32_t count = 0;
    
    bool is_type_ok = false; // check to see if this is a CFBag we know about
    if (descriptor->IsCFType())
    {
        ConstString type_name(valobj.GetTypeName());
        if (type_name == ConstString("__CFMutableBitVector") || type_name == ConstString("__CFBitVector") || type_name == ConstString("CFMutableBitVectorRef") || type_name == ConstString("CFBitVectorRef"))
        {
            if (valobj.IsPointerType())
                is_type_ok = true;
        }
    }
    
    if (is_type_ok == false)
        return false;
    
    Error error;
    count = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr+2*ptr_size, ptr_size, 0, error);
    if (error.Fail())
        return false;
    uint64_t num_bytes = count / 8 + ((count & 7) ? 1 : 0);
    addr_t data_ptr = process_sp->ReadPointerFromMemory(valobj_addr+2*ptr_size+2*ptr_size, error);
    if (error.Fail())
        return false;
    // make sure we do not try to read huge amounts of data
    if (num_bytes > 1024)
        num_bytes = 1024;
    DataBufferSP buffer_sp(new DataBufferHeap(num_bytes,0));
    num_bytes = process_sp->ReadMemory(data_ptr, buffer_sp->GetBytes(), num_bytes, error);
    if (error.Fail())
        return false;
    for (int byte_idx = 0; byte_idx < num_bytes-1; byte_idx++)
    {
        uint8_t byte = buffer_sp->GetBytes()[byte_idx];
        bool bit0 = (byte & 1) == 1;
        bool bit1 = (byte & 2) == 2;
        bool bit2 = (byte & 4) == 4;
        bool bit3 = (byte & 8) == 8;
        bool bit4 = (byte & 16) == 16;
        bool bit5 = (byte & 32) == 32;
        bool bit6 = (byte & 64) == 64;
        bool bit7 = (byte & 128) == 128;
        stream.Printf("%c%c%c%c %c%c%c%c ",
                      (bit7 ? '1' : '0'),
                      (bit6 ? '1' : '0'),
                      (bit5 ? '1' : '0'),
                      (bit4 ? '1' : '0'),
                      (bit3 ? '1' : '0'),
                      (bit2 ? '1' : '0'),
                      (bit1 ? '1' : '0'),
                      (bit0 ? '1' : '0'));
        count -= 8;
    }
    {
        // print the last byte ensuring we do not print spurious bits
        uint8_t byte = buffer_sp->GetBytes()[num_bytes-1];
        bool bit0 = (byte & 1) == 1;
        bool bit1 = (byte & 2) == 2;
        bool bit2 = (byte & 4) == 4;
        bool bit3 = (byte & 8) == 8;
        bool bit4 = (byte & 16) == 16;
        bool bit5 = (byte & 32) == 32;
        bool bit6 = (byte & 64) == 64;
        bool bit7 = (byte & 128) == 128;
        if (count)
        {
            stream.Printf("%c",bit7 ? '1' : '0');
            count -= 1;
        }
        if (count)
        {
            stream.Printf("%c",bit6 ? '1' : '0');
            count -= 1;
        }
        if (count)
        {
            stream.Printf("%c",bit5 ? '1' : '0');
            count -= 1;
        }
        if (count)
        {
            stream.Printf("%c",bit4 ? '1' : '0');
            count -= 1;
        }
        if (count)
        {
            stream.Printf("%c",bit3 ? '1' : '0');
            count -= 1;
        }
        if (count)
        {
            stream.Printf("%c",bit2 ? '1' : '0');
            count -= 1;
        }
        if (count)
        {
            stream.Printf("%c",bit1 ? '1' : '0');
            count -= 1;
        }
        if (count)
        {
            stream.Printf("%c",bit0 ? '1' : '0');
            count -= 1;
        }
    }
    return true;
}

bool
lldb_private::formatters::CFBinaryHeapSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    uint32_t count = 0;
    
    bool is_type_ok = false; // check to see if this is a CFBinaryHeap we know about
    if (descriptor->IsCFType())
    {
        ConstString type_name(valobj.GetTypeName());
        if (type_name == ConstString("__CFBinaryHeap") || type_name == ConstString("const struct __CFBinaryHeap"))
        {
            if (valobj.IsPointerType())
                is_type_ok = true;
        }
    }
    
    if (is_type_ok == false)
    {
        StackFrameSP frame_sp(valobj.GetFrameSP());
        if (!frame_sp)
            return false;
        ValueObjectSP count_sp;
        StreamString expr;
        expr.Printf("(int)CFBinaryHeapGetCount((void*)0x%" PRIx64 ")",valobj.GetPointerValue());
        if (process_sp->GetTarget().EvaluateExpression(expr.GetData(), frame_sp.get(), count_sp) != eExecutionCompleted)
            return false;
        if (!count_sp)
            return false;
        count = count_sp->GetValueAsUnsigned(0);
    }
    else
    {
        uint32_t offset = 2*ptr_size;
        Error error;
        count = process_sp->ReadUnsignedIntegerFromMemory(offset, 4, 0, error);
        if (error.Fail())
            return false;
    }
    stream.Printf("@\"%u item%s\"",
                  count,(count == 1 ? "" : "s"));
    return true;
}
