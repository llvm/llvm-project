//===-- Cocoa.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

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
lldb_private::formatters::NSBundleSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    if (!strcmp(class_name,"NSBundle"))
    {
        uint64_t offset = 5 * ptr_size;
        ClangASTType type(valobj.GetClangAST(),ClangASTContext::GetBuiltInType_objc_id(valobj.GetClangAST()));
        ValueObjectSP text(valobj.GetSyntheticChildAtOffset(offset, type, true));

        StreamString summary_stream;
        bool was_nsstring_ok = NSStringSummaryProvider(*text.get(), summary_stream);
        if (was_nsstring_ok && summary_stream.GetSize() > 0)
        {
            stream.Printf("%s",summary_stream.GetData());
            return true;
        }
    }
    // this is either an unknown subclass or an NSBundle that comes from [NSBundle mainBundle]
    // which is encoded differently and needs to be handled by running code
    return ExtractSummaryFromObjCExpression(valobj, "NSString*", "bundlePath", stream);
}

bool
lldb_private::formatters::NSTimeZoneSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    if (!strcmp(class_name,"__NSTimeZone"))
    {
        uint64_t offset = ptr_size;
        ClangASTType type(valobj.GetClangAST(),valobj.GetClangType());
        ValueObjectSP text(valobj.GetSyntheticChildAtOffset(offset, type, true));
        StreamString summary_stream;
        bool was_nsstring_ok = NSStringSummaryProvider(*text.get(), summary_stream);
        if (was_nsstring_ok && summary_stream.GetSize() > 0)
        {
            stream.Printf("%s",summary_stream.GetData());
            return true;
        }
    }
    return ExtractSummaryFromObjCExpression(valobj, "NSString*", "name", stream);
}

bool
lldb_private::formatters::NSNotificationSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    if (!strcmp(class_name,"NSConcreteNotification"))
    {
        uint64_t offset = ptr_size;
        ClangASTType type(valobj.GetClangAST(),valobj.GetClangType());
        ValueObjectSP text(valobj.GetSyntheticChildAtOffset(offset, type, true));
        StreamString summary_stream;
        bool was_nsstring_ok = NSStringSummaryProvider(*text.get(), summary_stream);
        if (was_nsstring_ok && summary_stream.GetSize() > 0)
        {
            stream.Printf("%s",summary_stream.GetData());
            return true;
        }
    }
    // this is either an unknown subclass or an NSBundle that comes from [NSBundle mainBundle]
    // which is encoded differently and needs to be handled by running code
    return ExtractSummaryFromObjCExpression(valobj, "NSString*", "name", stream);
}

bool
lldb_private::formatters::NSMachPortSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    uint64_t port_number = 0;
    
    do
    {
        if (!strcmp(class_name,"NSMachPort"))
        {
            uint64_t offset = (ptr_size == 4 ? 12 : 20);
            Error error;
            port_number = process_sp->ReadUnsignedIntegerFromMemory(offset+valobj_addr, 4, 0, error);
            if (error.Success())
                break;
        }
        if (!ExtractValueFromObjCExpression(valobj, "int", "machPort", port_number))
            return false;
    } while (false);
    
    stream.Printf("mach port: %u",(uint32_t)(port_number & 0x00000000FFFFFFFF));
    return true;
}

bool
lldb_private::formatters::NSIndexSetSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    uint64_t count = 0;
    
    do {
        if (!strcmp(class_name,"NSIndexSet") || !strcmp(class_name,"NSMutableIndexSet"))
        {
            Error error;
            uint32_t mode = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr+ptr_size, 4, 0, error);
            if (error.Fail())
                return false;
            // this means the set is empty - count = 0
            if ((mode & 1) == 1)
            {
                count = 0;
                break;
            }
            if ((mode & 2) == 2)
                mode = 1; // this means the set only has one range
            else
                mode = 2; // this means the set has multiple ranges
            if (mode == 1)
            {
                count = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr+3*ptr_size, ptr_size, 0, error);
                if (error.Fail())
                    return false;
            }
            else
            {
                // read a pointer to the data at 2*ptr_size
                count = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr+2*ptr_size, ptr_size, 0, error);
                if (error.Fail())
                    return false;
                // read the data at 2*ptr_size from the first location
                count = process_sp->ReadUnsignedIntegerFromMemory(count+2*ptr_size, ptr_size, 0, error);
                if (error.Fail())
                    return false;
            }
        }
        else
        {
            if (!ExtractValueFromObjCExpression(valobj, "unsigned long long int", "count", count))
                return false;
        }
    }  while (false);
    stream.Printf("%" PRIu64 " index%s",
                  count,
                  (count == 1 ? "" : "es"));
    return true;
}

bool
lldb_private::formatters::NSNumberSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    if (!strcmp(class_name,"NSNumber") || !strcmp(class_name,"__NSCFNumber"))
    {
        uint64_t value = 0;
        uint64_t i_bits = 0;
        if (descriptor->GetTaggedPointerInfo(&i_bits,&value))
        {
            switch (i_bits)
            {
                case 0:
                    stream.Printf("(char)%hhd",(char)value);
                    break;
                case 1:
                case 4:
                    stream.Printf("(short)%hd",(short)value);
                    break;
                case 2:
                case 8:
                    stream.Printf("(int)%d",(int)value);
                    break;
                case 3:
                case 12:
                    stream.Printf("(long)%" PRId64,value);
                    break;
                default:
                    stream.Printf("unexpected value:(info=%" PRIu64 ", value=%" PRIu64,i_bits,value);
                    break;
            }
            return true;
        }
        else
        {
            Error error;
            uint8_t data_type = (process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, 1, 0, error) & 0x1F);
            uint64_t data_location = valobj_addr + 2*ptr_size;
            uint64_t value = 0;
            if (error.Fail())
                return false;
            switch (data_type)
            {
                case 1: // 0B00001
                    value = process_sp->ReadUnsignedIntegerFromMemory(data_location, 1, 0, error);
                    if (error.Fail())
                        return false;
                    stream.Printf("(char)%hhd",(char)value);
                    break;
                case 2: // 0B0010
                    value = process_sp->ReadUnsignedIntegerFromMemory(data_location, 2, 0, error);
                    if (error.Fail())
                        return false;
                    stream.Printf("(short)%hd",(short)value);
                    break;
                case 3: // 0B0011
                    value = process_sp->ReadUnsignedIntegerFromMemory(data_location, 4, 0, error);
                    if (error.Fail())
                        return false;
                    stream.Printf("(int)%d",(int)value);
                    break;
                case 17: // 0B10001
                    data_location += 8;
                case 4: // 0B0100
                    value = process_sp->ReadUnsignedIntegerFromMemory(data_location, 8, 0, error);
                    if (error.Fail())
                        return false;
                    stream.Printf("(long)%" PRId64,value);
                    break;
                case 5: // 0B0101
                {
                    uint32_t flt_as_int = process_sp->ReadUnsignedIntegerFromMemory(data_location, 4, 0, error);
                    if (error.Fail())
                        return false;
                    float flt_value = *((float*)&flt_as_int);
                    stream.Printf("(float)%f",flt_value);
                    break;
                }
                case 6: // 0B0110
                {
                    uint64_t dbl_as_lng = process_sp->ReadUnsignedIntegerFromMemory(data_location, 8, 0, error);
                    if (error.Fail())
                        return false;
                    double dbl_value = *((double*)&dbl_as_lng);
                    stream.Printf("(double)%g",dbl_value);
                    break;
                }
                default:
                    stream.Printf("unexpected value: dt=%d",data_type);
                    break;
            }
            return true;
        }
    }
    else
    {
        return ExtractSummaryFromObjCExpression(valobj, "NSString*", "stringValue", stream);
    }
}

bool
lldb_private::formatters::NSURLSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    if (strcmp(class_name, "NSURL") == 0)
    {
        uint64_t offset_text = ptr_size + ptr_size + 8; // ISA + pointer + 8 bytes of data (even on 32bit)
        uint64_t offset_base = offset_text + ptr_size;
        ClangASTType type(valobj.GetClangAST(),valobj.GetClangType());
        ValueObjectSP text(valobj.GetSyntheticChildAtOffset(offset_text, type, true));
        ValueObjectSP base(valobj.GetSyntheticChildAtOffset(offset_base, type, true));
        if (!text)
            return false;
        if (text->GetValueAsUnsigned(0) == 0)
            return false;
        StreamString summary;
        if (!NSStringSummaryProvider(*text, summary))
            return false;
        if (base && base->GetValueAsUnsigned(0))
        {
            if (summary.GetSize() > 0)
                summary.GetString().resize(summary.GetSize()-1);
            summary.Printf(" -- ");
            StreamString base_summary;
            if (NSURLSummaryProvider(*base, base_summary) && base_summary.GetSize() > 0)
                summary.Printf("%s",base_summary.GetSize() > 2 ? base_summary.GetData() + 2 : base_summary.GetData());
        }
        if (summary.GetSize())
        {
            stream.Printf("%s",summary.GetData());
            return true;
        }
    }
    else
    {
        return ExtractSummaryFromObjCExpression(valobj, "NSString*", "description", stream);
    }
    return false;
}

bool
lldb_private::formatters::NSDateSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    uint64_t date_value_bits = 0;
    double date_value = 0.0;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    if (strcmp(class_name,"NSDate") == 0 ||
        strcmp(class_name,"__NSDate") == 0 ||
        strcmp(class_name,"__NSTaggedDate") == 0)
    {
        uint64_t info_bits=0,value_bits = 0;
        if (descriptor->GetTaggedPointerInfo(&info_bits,&value_bits))
        {
            date_value_bits = ((value_bits << 8) | (info_bits << 4));
            date_value = *((double*)&date_value_bits);
        }
        else
        {
            Error error;
            date_value_bits = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr+ptr_size, 8, 0, error);
            date_value = *((double*)&date_value_bits);
            if (error.Fail())
                return false;
        }
    }
    else if (!strcmp(class_name,"NSCalendarDate"))
    {
        Error error;
        date_value_bits = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr+2*ptr_size, 8, 0, error);
        date_value = *((double*)&date_value_bits);
        if (error.Fail())
            return false;
    }
    else
    {
        if (ExtractValueFromObjCExpression(valobj, "NSTimeInterval", "ExtractValueFromObjCExpression", date_value_bits) == false)
            return false;
        date_value = *((double*)&date_value_bits);
    }
    if (date_value == -63114076800)
    {
        stream.Printf("0001-12-30 00:00:00 +0000");
        return true;
    }
    // this snippet of code assumes that time_t == seconds since Jan-1-1970
    // this is generally true and POSIXly happy, but might break if a library
    // vendor decides to get creative
    time_t epoch = GetOSXEpoch();
    epoch = epoch + (time_t)date_value;
    tm *tm_date = localtime(&epoch);
    if (!tm_date)
        return false;
    std::string buffer(1024,0);
    if (strftime (&buffer[0], 1023, "%Z", tm_date) == 0)
        return false;
    stream.Printf("%04d-%02d-%02d %02d:%02d:%02d %s", tm_date->tm_year+1900, tm_date->tm_mon+1, tm_date->tm_mday, tm_date->tm_hour, tm_date->tm_min, tm_date->tm_sec, buffer.c_str());
    return true;
}
