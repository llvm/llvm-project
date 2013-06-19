//===-- CXXFormatterFunctions.cpp---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/DataFormatters/CXXFormatterFunctions.h"

#include "llvm/Support/ConvertUTF.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

#include <algorithm>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

bool
lldb_private::formatters::ExtractValueFromObjCExpression (ValueObject &valobj,
                                                          const char* target_type,
                                                          const char* selector,
                                                          uint64_t &value)
{
    if (!target_type || !*target_type)
        return false;
    if (!selector || !*selector)
        return false;
    StreamString expr;
    expr.Printf("(%s)[(id)0x%" PRIx64 " %s]",target_type,valobj.GetPointerValue(),selector);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = exe_ctx.GetFramePtr();
    if (!target || !stack_frame)
        return false;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               result_sp,
                               options);
    if (!result_sp)
        return false;
    value = result_sp->GetValueAsUnsigned(0);
    return true;
}

bool
lldb_private::formatters::ExtractSummaryFromObjCExpression (ValueObject &valobj,
                                                            const char* target_type,
                                                            const char* selector,
                                                            Stream &stream)
{
    if (!target_type || !*target_type)
        return false;
    if (!selector || !*selector)
        return false;
    StreamString expr;
    expr.Printf("(%s)[(id)0x%" PRIx64 " %s]",target_type,valobj.GetPointerValue(),selector);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = exe_ctx.GetFramePtr();
    if (!target || !stack_frame)
        return false;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true)
    .SetUseDynamic(lldb::eDynamicCanRunTarget);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               result_sp,
                               options);
    if (!result_sp)
        return false;
    stream.Printf("%s",result_sp->GetSummaryAsCString());
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::CallSelectorOnObject (ValueObject &valobj,
                                                const char* return_type,
                                                const char* selector,
                                                uint64_t index)
{
    lldb::ValueObjectSP valobj_sp;
    if (!return_type || !*return_type)
        return valobj_sp;
    if (!selector || !*selector)
        return valobj_sp;
    StreamString expr_path_stream;
    valobj.GetExpressionPath(expr_path_stream, false);
    StreamString expr;
    expr.Printf("(%s)[%s %s:%" PRId64 "]",return_type,expr_path_stream.GetData(),selector,index);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = exe_ctx.GetFramePtr();
    if (!target || !stack_frame)
        return valobj_sp;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true)
    .SetUseDynamic(lldb::eDynamicCanRunTarget);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               valobj_sp,
                               options);
    return valobj_sp;
}

lldb::ValueObjectSP
lldb_private::formatters::CallSelectorOnObject (ValueObject &valobj,
                                                const char* return_type,
                                                const char* selector,
                                                const char* key)
{
    lldb::ValueObjectSP valobj_sp;
    if (!return_type || !*return_type)
        return valobj_sp;
    if (!selector || !*selector)
        return valobj_sp;
    if (!key || !*key)
        return valobj_sp;
    StreamString expr_path_stream;
    valobj.GetExpressionPath(expr_path_stream, false);
    StreamString expr;
    expr.Printf("(%s)[%s %s:%s]",return_type,expr_path_stream.GetData(),selector,key);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = exe_ctx.GetFramePtr();
    if (!target || !stack_frame)
        return valobj_sp;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true)
    .SetUseDynamic(lldb::eDynamicCanRunTarget);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               valobj_sp,
                               options);
    return valobj_sp;
}

// use this call if you already have an LLDB-side buffer for the data
template<typename SourceDataType>
static bool
DumpUTFBufferToStream (ConversionResult (*ConvertFunction) (const SourceDataType**,
                                                            const SourceDataType*,
                                                            UTF8**,
                                                            UTF8*,
                                                            ConversionFlags),
                       DataExtractor& data,
                       Stream& stream,
                       char prefix_token = '@',
                       char quote = '"',
                       uint32_t sourceSize = 0)
{
    if (prefix_token != 0)
        stream.Printf("%c",prefix_token);
    if (quote != 0)
        stream.Printf("%c",quote);
    if (data.GetByteSize() && data.GetDataStart() && data.GetDataEnd())
    {
        const int bufferSPSize = data.GetByteSize();
        if (sourceSize == 0)
        {
            const int origin_encoding = 8*sizeof(SourceDataType);
            sourceSize = bufferSPSize/(origin_encoding / 4);
        }
        
        SourceDataType *data_ptr = (SourceDataType*)data.GetDataStart();
        SourceDataType *data_end_ptr = data_ptr + sourceSize;
        
        while (data_ptr < data_end_ptr)
        {
            if (!*data_ptr)
            {
                data_end_ptr = data_ptr;
                break;
            }
            data_ptr++;
        }
        
        data_ptr = (SourceDataType*)data.GetDataStart();
        
        lldb::DataBufferSP utf8_data_buffer_sp;
        UTF8* utf8_data_ptr = nullptr;
        UTF8* utf8_data_end_ptr = nullptr;
        
        if (ConvertFunction)
        {
            utf8_data_buffer_sp.reset(new DataBufferHeap(4*bufferSPSize,0));
            utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes();
            utf8_data_end_ptr = utf8_data_ptr + utf8_data_buffer_sp->GetByteSize();
            ConvertFunction ( (const SourceDataType**)&data_ptr, data_end_ptr, &utf8_data_ptr, utf8_data_end_ptr, lenientConversion );
            utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes(); // needed because the ConvertFunction will change the value of the data_ptr
        }
        else
        {
            // just copy the pointers - the cast is necessary to make the compiler happy
            // but this should only happen if we are reading UTF8 data
            utf8_data_ptr = (UTF8*)data_ptr;
            utf8_data_end_ptr = (UTF8*)data_end_ptr;
        }
        
        // since we tend to accept partial data (and even partially malformed data)
        // we might end up with no NULL terminator before the end_ptr
        // hence we need to take a slower route and ensure we stay within boundaries
        for (;utf8_data_ptr != utf8_data_end_ptr; utf8_data_ptr++)
        {
            if (!*utf8_data_ptr)
                break;
            stream.Printf("%c",*utf8_data_ptr);
        }
    }
    if (quote != 0)
        stream.Printf("%c",quote);
    return true;
}

template<typename SourceDataType>
class ReadUTFBufferAndDumpToStreamOptions
{
public:
    typedef ConversionResult (*ConvertFunctionType) (const SourceDataType**,
                                                     const SourceDataType*,
                                                     UTF8**,
                                                     UTF8*,
                                                     ConversionFlags);
    
    ReadUTFBufferAndDumpToStreamOptions () :
    m_conversion_function(NULL),
    m_location(0),
    m_process_sp(),
    m_stream(NULL),
    m_prefix_token('@'),
    m_quote('"'),
    m_source_size(0),
    m_needs_zero_termination(true)
    {
    }
    
    ReadUTFBufferAndDumpToStreamOptions&
    SetConversionFunction (ConvertFunctionType f)
    {
        m_conversion_function = f;
        return *this;
    }
    
    ConvertFunctionType
    GetConversionFunction () const
    {
        return m_conversion_function;
    }
    
    ReadUTFBufferAndDumpToStreamOptions&
    SetLocation (uint64_t l)
    {
        m_location = l;
        return *this;
    }
    
    uint64_t
    GetLocation () const
    {
        return m_location;
    }
    
    ReadUTFBufferAndDumpToStreamOptions&
    SetProcessSP (ProcessSP p)
    {
        m_process_sp = p;
        return *this;
    }
    
    ProcessSP
    GetProcessSP () const
    {
        return m_process_sp;
    }
    
    ReadUTFBufferAndDumpToStreamOptions&
    SetStream (Stream* s)
    {
        m_stream = s;
        return *this;
    }
    
    Stream*
    GetStream () const
    {
        return m_stream;
    }
    
    ReadUTFBufferAndDumpToStreamOptions&
    SetPrefixToken (char p)
    {
        m_prefix_token = p;
        return *this;
    }
    
    char
    GetPrefixToken () const
    {
        return m_prefix_token;
    }
    
    ReadUTFBufferAndDumpToStreamOptions&
    SetQuote (char q)
    {
        m_quote = q;
        return *this;
    }
    
    char
    GetQuote () const
    {
        return m_quote;
    }
    
    ReadUTFBufferAndDumpToStreamOptions&
    SetSourceSize (uint32_t s)
    {
        m_source_size = s;
        return *this;
    }
    
    uint32_t
    GetSourceSize () const
    {
        return m_source_size;
    }
    
    ReadUTFBufferAndDumpToStreamOptions&
    SetNeedsZeroTermination (bool z)
    {
        m_needs_zero_termination = z;
        return *this;
    }
    
    bool
    GetNeedsZeroTermination () const
    {
        return m_needs_zero_termination;
    }
    
private:
    ConvertFunctionType m_conversion_function;
    uint64_t m_location;
    ProcessSP m_process_sp;
    Stream* m_stream;
    char m_prefix_token;
    char m_quote;
    uint32_t m_source_size;
    bool m_needs_zero_termination;
};

template<typename SourceDataType>
static bool
ReadUTFBufferAndDumpToStream (const ReadUTFBufferAndDumpToStreamOptions<SourceDataType>& options)
{
    if (options.GetLocation() == 0 || options.GetLocation() == LLDB_INVALID_ADDRESS)
        return false;
    
    ProcessSP process_sp(options.GetProcessSP());
    
    if (!process_sp)
        return false;

    const int type_width = sizeof(SourceDataType);
    const int origin_encoding = 8 * type_width ;
    if (origin_encoding != 8 && origin_encoding != 16 && origin_encoding != 32)
        return false;
    // if not UTF8, I need a conversion function to return proper UTF8
    if (origin_encoding != 8 && !options.GetConversionFunction())
        return false;
    
    if (!options.GetStream())
        return false;

    uint32_t sourceSize = options.GetSourceSize();
    bool needs_zero_terminator = options.GetNeedsZeroTermination();
    
    if (!sourceSize)
    {
        sourceSize = process_sp->GetTarget().GetMaximumSizeOfStringSummary();
        needs_zero_terminator = true;
    }
    else
        sourceSize = std::min(sourceSize,process_sp->GetTarget().GetMaximumSizeOfStringSummary());
    
    const int bufferSPSize = sourceSize * type_width;

    lldb::DataBufferSP buffer_sp(new DataBufferHeap(bufferSPSize,0));
    
    if (!buffer_sp->GetBytes())
        return false;
    
    Error error;
    char *buffer = reinterpret_cast<char *>(buffer_sp->GetBytes()); 

    size_t data_read = 0;
    if (needs_zero_terminator)
        data_read = process_sp->ReadStringFromMemory(options.GetLocation(), buffer, bufferSPSize, error, type_width);
    else
        data_read = process_sp->ReadMemoryFromInferior(options.GetLocation(), (char*)buffer_sp->GetBytes(), bufferSPSize, error);

    if (error.Fail() || data_read == 0)
    {
        options.GetStream()->Printf("unable to read data");
        return true;
    }
    
    DataExtractor data(buffer_sp, process_sp->GetByteOrder(), process_sp->GetAddressByteSize());
    
    return DumpUTFBufferToStream(options.GetConversionFunction(), data, *options.GetStream(), options.GetPrefixToken(), options.GetQuote(), sourceSize);
}

bool
lldb_private::formatters::Char16StringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    ReadUTFBufferAndDumpToStreamOptions<UTF16> options;
    options.SetLocation(valobj_addr);
    options.SetConversionFunction(ConvertUTF16toUTF8);
    options.SetProcessSP(process_sp);
    options.SetStream(&stream);
    options.SetPrefixToken('u');
    
    if (!ReadUTFBufferAndDumpToStream(options))
    {
        stream.Printf("Summary Unavailable");
        return true;
    }

    return true;
}

bool
lldb_private::formatters::Char32StringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    ReadUTFBufferAndDumpToStreamOptions<UTF32> options;
    options.SetLocation(valobj_addr);
    options.SetConversionFunction(ConvertUTF32toUTF8);
    options.SetProcessSP(process_sp);
    options.SetStream(&stream);
    options.SetPrefixToken('U');
    
    if (!ReadUTFBufferAndDumpToStream(options))
    {
        stream.Printf("Summary Unavailable");
        return true;
    }
    
    return true;
}

bool
lldb_private::formatters::WCharStringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;

    lldb::addr_t data_addr = 0;
    
    if (valobj.IsPointerType())
        data_addr = valobj.GetValueAsUnsigned(0);
    else if (valobj.IsArrayType())
        data_addr = valobj.GetAddressOf();

    if (data_addr == 0 || data_addr == LLDB_INVALID_ADDRESS)
        return false;

    clang::ASTContext* ast = valobj.GetClangAST();

    if (!ast)
        return false;

    uint32_t wchar_size = ClangASTType::GetClangTypeBitWidth(ast, ClangASTType::GetBasicType(ast, lldb::eBasicTypeWChar).GetOpaqueQualType());

    switch (wchar_size)
    {
        case 8:
        {
            // utf 8
            
            ReadUTFBufferAndDumpToStreamOptions<UTF8> options;
            options.SetLocation(data_addr);
            options.SetConversionFunction(nullptr);
            options.SetProcessSP(process_sp);
            options.SetStream(&stream);
            options.SetPrefixToken('L');

            return ReadUTFBufferAndDumpToStream(options);
        }
        case 16:
        {
            // utf 16
            ReadUTFBufferAndDumpToStreamOptions<UTF16> options;
            options.SetLocation(data_addr);
            options.SetConversionFunction(ConvertUTF16toUTF8);
            options.SetProcessSP(process_sp);
            options.SetStream(&stream);
            options.SetPrefixToken('L');
            
            return ReadUTFBufferAndDumpToStream(options);
        }
        case 32:
        {
            // utf 32
            ReadUTFBufferAndDumpToStreamOptions<UTF32> options;
            options.SetLocation(data_addr);
            options.SetConversionFunction(ConvertUTF32toUTF8);
            options.SetProcessSP(process_sp);
            options.SetStream(&stream);
            options.SetPrefixToken('L');
            
            return ReadUTFBufferAndDumpToStream(options);
        }
        default:
            stream.Printf("size for wchar_t is not valid");
            return true;
    }
    return true;
}

bool
lldb_private::formatters::Char16SummaryProvider (ValueObject& valobj, Stream& stream)
{
    DataExtractor data;
    valobj.GetData(data);
    
    std::string value;
    valobj.GetValueAsCString(lldb::eFormatUnicode16, value);
    if (!value.empty())
        stream.Printf("%s ", value.c_str());

    return DumpUTFBufferToStream<UTF16>(ConvertUTF16toUTF8,data,stream, 'u','\'',1);
}

bool
lldb_private::formatters::Char32SummaryProvider (ValueObject& valobj, Stream& stream)
{
    DataExtractor data;
    valobj.GetData(data);
    
    std::string value;
    valobj.GetValueAsCString(lldb::eFormatUnicode32, value);
    if (!value.empty())
        stream.Printf("%s ", value.c_str());
    
    return DumpUTFBufferToStream<UTF32>(ConvertUTF32toUTF8,data,stream, 'U','\'',1);
}

bool
lldb_private::formatters::WCharSummaryProvider (ValueObject& valobj, Stream& stream)
{
    DataExtractor data;
    valobj.GetData(data);
    
    clang::ASTContext* ast = valobj.GetClangAST();
    
    if (!ast)
        return false;
    
    std::string value;
    
    uint32_t wchar_size = ClangASTType::GetClangTypeBitWidth(ast, ClangASTType::GetBasicType(ast, lldb::eBasicTypeWChar).GetOpaqueQualType());
    
    switch (wchar_size)
    {
        case 8:
            // utf 8
            valobj.GetValueAsCString(lldb::eFormatChar, value);
            if (!value.empty())
                stream.Printf("%s ", value.c_str());
            return DumpUTFBufferToStream<UTF8>(nullptr,
                                               data,
                                               stream,
                                               'L',
                                               '\'',
                                               1);
        case 16:
            // utf 16
            valobj.GetValueAsCString(lldb::eFormatUnicode16, value);
            if (!value.empty())
                stream.Printf("%s ", value.c_str());
            return DumpUTFBufferToStream<UTF16>(ConvertUTF16toUTF8,
                                                data,
                                                stream,
                                                'L',
                                                '\'',
                                                1);
        case 32:
            // utf 32
            valobj.GetValueAsCString(lldb::eFormatUnicode32, value);
            if (!value.empty())
                stream.Printf("%s ", value.c_str());
            return DumpUTFBufferToStream<UTF32>(ConvertUTF32toUTF8,
                                                data,
                                                stream,
                                                'L',
                                                '\'',
                                                1);
        default:
            stream.Printf("size for wchar_t is not valid");
            return true;
    }
    return true;
}

// the field layout in a libc++ string (cap, side, data or data, size, cap)
enum LibcxxStringLayoutMode
{
    eLibcxxStringLayoutModeCSD = 0,
    eLibcxxStringLayoutModeDSC = 1,
    eLibcxxStringLayoutModeInvalid = 0xffff
};

// this function abstracts away the layout and mode details of a libc++ string
// and returns the address of the data and the size ready for callers to consume
static bool
ExtractLibcxxStringInfo (ValueObject& valobj,
                         ValueObjectSP &location_sp,
                         uint64_t& size)
{
    ValueObjectSP D(valobj.GetChildAtIndexPath({0,0,0,0}));
    if (!D)
        return false;
    
    ValueObjectSP layout_decider(D->GetChildAtIndexPath({0,0}));
    
    // this child should exist
    if (!layout_decider)
        return false;
    
    ConstString g_data_name("__data_");
    ConstString g_size_name("__size_");
    bool short_mode = false; // this means the string is in short-mode and the data is stored inline
    LibcxxStringLayoutMode layout = (layout_decider->GetName() == g_data_name) ? eLibcxxStringLayoutModeDSC : eLibcxxStringLayoutModeCSD;
    uint64_t size_mode_value = 0;
    
    if (layout == eLibcxxStringLayoutModeDSC)
    {
        ValueObjectSP size_mode(D->GetChildAtIndexPath({1,1,0}));
        if (!size_mode)
            return false;
        
        if (size_mode->GetName() != g_size_name)
        {
            // we are hitting the padding structure, move along
            size_mode = D->GetChildAtIndexPath({1,1,1});
            if (!size_mode)
                return false;
        }
        
        size_mode_value = (size_mode->GetValueAsUnsigned(0));
        short_mode = ((size_mode_value & 0x80) == 0);
    }
    else
    {
        ValueObjectSP size_mode(D->GetChildAtIndexPath({1,0,0}));
        if (!size_mode)
            return false;
        
        size_mode_value = (size_mode->GetValueAsUnsigned(0));
        short_mode = ((size_mode_value & 1) == 0);
    }
    
    if (short_mode)
    {
        ValueObjectSP s(D->GetChildAtIndex(1, true));
        if (!s)
            return false;
        location_sp = s->GetChildAtIndex((layout == eLibcxxStringLayoutModeDSC) ? 0 : 1, true);
        size = (layout == eLibcxxStringLayoutModeDSC) ? size_mode_value : ((size_mode_value >> 1) % 256);
        return (location_sp.get() != nullptr);
    }
    else
    {
        ValueObjectSP l(D->GetChildAtIndex(0, true));
        if (!l)
            return false;
        // we can use the layout_decider object as the data pointer
        location_sp = (layout == eLibcxxStringLayoutModeDSC) ? layout_decider : l->GetChildAtIndex(2, true);
        ValueObjectSP size_vo(l->GetChildAtIndex(1, true));
        if (!size_vo || !location_sp)
            return false;
        size = size_vo->GetValueAsUnsigned(0);
        return true;
    }
}

bool
lldb_private::formatters::LibcxxWStringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    uint64_t size = 0;
    ValueObjectSP location_sp((ValueObject*)nullptr);
    if (!ExtractLibcxxStringInfo(valobj, location_sp, size))
        return false;
    if (size == 0)
    {
        stream.Printf("L\"\"");
        return true;
    }   
    if (!location_sp)
        return false;
    return WCharStringSummaryProvider(*location_sp.get(), stream);
}

bool
lldb_private::formatters::LibcxxStringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    uint64_t size = 0;
    ValueObjectSP location_sp((ValueObject*)nullptr);
    if (!ExtractLibcxxStringInfo(valobj, location_sp, size))
        return false;
    if (size == 0)
    {
        stream.Printf("\"\"");
        return true;
    }
    if (!location_sp)
        return false;
    Error error;
    if (location_sp->ReadPointedString(stream,
                                       error,
                                       0, // max length is decided by the settings
                                       false) == 0) // do not honor array (terminates on first 0 byte even for a char[])
        stream.Printf("\"\""); // if nothing was read, print an empty string
    return error.Success();
}

bool
lldb_private::formatters::ObjCClassSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptorFromISA(valobj.GetValueAsUnsigned(0)));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;

    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    stream.Printf("%s",class_name);
    return true;
}

class ObjCClassSyntheticChildrenFrontEnd : public SyntheticChildrenFrontEnd
{
public:
    ObjCClassSyntheticChildrenFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp.get())
    {
    }
    
    virtual size_t
    CalculateNumChildren ()
    {
        return 0;
    }
    
    virtual lldb::ValueObjectSP
    GetChildAtIndex (size_t idx)
    {
        return lldb::ValueObjectSP();
    }
    
    virtual bool
    Update()
    {
        return false;
    }
    
    virtual bool
    MightHaveChildren ()
    {
        return false;
    }
    
    virtual size_t
    GetIndexOfChildWithName (const ConstString &name)
    {
        return UINT32_MAX;
    }
    
    virtual
    ~ObjCClassSyntheticChildrenFrontEnd ()
    {
    }
};

SyntheticChildrenFrontEnd*
lldb_private::formatters::ObjCClassSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    return new ObjCClassSyntheticChildrenFrontEnd(valobj_sp);
}

template<bool needs_at>
bool
lldb_private::formatters::NSDataSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    bool is_64bit = (process_sp->GetAddressByteSize() == 8);
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    uint64_t value = 0;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    if (!strcmp(class_name,"NSConcreteData") ||
        !strcmp(class_name,"NSConcreteMutableData") ||
        !strcmp(class_name,"__NSCFData"))
    {
        uint32_t offset = (is_64bit ? 16 : 8);
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + offset, is_64bit ? 8 : 4, 0, error);
        if (error.Fail())
            return false;
    }
    else
    {
        if (!ExtractValueFromObjCExpression(valobj, "int", "length", value))
            return false;
    }
    
    stream.Printf("%s%" PRIu64 " byte%s%s",
                  (needs_at ? "@\"" : ""),
                  value,
                  (value != 1 ? "s" : ""),
                  (needs_at ? "\"" : ""));
    
    return true;
}

static bool
ReadAsciiBufferAndDumpToStream (lldb::addr_t location,
                                lldb::ProcessSP& process_sp,
                                Stream& dest,
                                uint32_t size = 0,
                                Error* error = NULL,
                                size_t *data_read = NULL,
                                char prefix_token = '@',
                                char quote = '"')
{
    Error my_error;
    size_t my_data_read;
    if (!process_sp || location == 0)
        return false;
    
    if (!size)
        size = process_sp->GetTarget().GetMaximumSizeOfStringSummary();
    else
        size = std::min(size,process_sp->GetTarget().GetMaximumSizeOfStringSummary());
    
    lldb::DataBufferSP buffer_sp(new DataBufferHeap(size,0));
    
    my_data_read = process_sp->ReadCStringFromMemory(location, (char*)buffer_sp->GetBytes(), size, my_error);

    if (error)
        *error = my_error;
    if (data_read)
        *data_read = my_data_read;
    
    if (my_error.Fail())
        return false;
    
    dest.Printf("%c%c",prefix_token,quote);
    
    if (my_data_read)
        dest.Printf("%s",(char*)buffer_sp->GetBytes());
    
    dest.Printf("%c",quote);
    
    return true;
}

#ifdef WANT_DEEP_PRINT
struct lldb__notInlineMutable {
    void *buffer;
    signed long length;
    signed long capacity;
    unsigned int hasGap:1;
    unsigned int isFixedCapacity:1;
    unsigned int isExternalMutable:1;
    unsigned int capacityProvidedExternally:1;
#if __LP64__
    unsigned long desiredCapacity:60;
#else
    unsigned long desiredCapacity:28;
#endif
    void* contentsAllocator;
};

struct lldb__CFString {
    uintptr_t _cfisa;
    uint8_t _cfinfo[4];
    uint32_t _rc;
    union {
        struct __inline1 {
            signed long length;
        } inline1;
        struct __notInlineImmutable1 {
            void *buffer;
            signed long length;
            void* contentsDeallocator;
        } notInlineImmutable1;
        struct __notInlineImmutable2 {
            void *buffer;
            void* contentsDeallocator;
        } notInlineImmutable2;
        struct lldb__notInlineMutable notInlineMutable;
    } variants;
};
#endif

bool
lldb_private::formatters::NSStringSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    uint64_t info_bits_location = valobj_addr + ptr_size;
    if (process_sp->GetByteOrder() != lldb::eByteOrderLittle)
        info_bits_location += 3;
        
    Error error;
    
    uint8_t info_bits = process_sp->ReadUnsignedIntegerFromMemory(info_bits_location, 1, 0, error);
    if (error.Fail())
        return false;
    
    bool is_mutable = (info_bits & 1) == 1;
    bool is_inline = (info_bits & 0x60) == 0;
    bool has_explicit_length = (info_bits & (1 | 4)) != 4;
    bool is_unicode = (info_bits & 0x10) == 0x10;
    bool is_special = strcmp(class_name,"NSPathStore2") == 0;
    bool has_null = (info_bits & 8) == 8;
    
    size_t explicit_length = 0;
    if (!has_null && has_explicit_length && !is_special)
    {
        lldb::addr_t explicit_length_offset = 2*ptr_size;
        if (is_mutable and not is_inline)
            explicit_length_offset = explicit_length_offset + ptr_size; //  notInlineMutable.length;
        else if (is_inline)
            explicit_length = explicit_length + 0; // inline1.length;
        else if (not is_inline and not is_mutable)
            explicit_length_offset = explicit_length_offset + ptr_size; // notInlineImmutable1.length;
        else
            explicit_length_offset = 0;

        if (explicit_length_offset)
        {
            explicit_length_offset = valobj_addr + explicit_length_offset;
            explicit_length = process_sp->ReadUnsignedIntegerFromMemory(explicit_length_offset, 4, 0, error);
        }
    }
    
#ifdef WANT_DEEP_PRINT
    lldb__CFString my_string_data;
    process_sp->ReadMemory(valobj_addr, &my_string_data, sizeof(lldb__CFString),error);
    
    printf(R"(
           __CFString my_string_data = {
           uintptr_t _cfisa = %lu
           uint8_t _cfinfo[4] = %c%c%c%c
           uint32_t _rc = %d
           union {
               struct __inline1 {
                   signed long length = %ld
               } inline1;
               struct __notInlineImmutable1 {
                   void *buffer = %p
                   signed long length = %ld
                   void* contentsDeallocator = %p
               } notInlineImmutable1;
               struct __notInlineImmutable2 {
                   void *buffer = %p
                   void* contentsDeallocator = %p
               } notInlineImmutable2;
               struct __notInlineMutable notInlineMutable {
                   void *buffer = %p
                   signed long length = %ld
                   signed long capacity = %ld
                   unsigned int hasGap:1 = %d
                   unsigned int isFixedCapacity:1 = %d
                   unsigned int isExternalMutable:1 = %d
                   unsigned int capacityProvidedExternally:1 = %d
#if __LP64__
                   unsigned long desiredCapacity:60 = %lu
#else
                   unsigned long desiredCapacity:28 = %lu
#endif
                   void* contentsAllocator = %p
               }
           } variants; ==> (M:%dI:%dL:%zuU:%dS:%dN:%d)
           };\n)",
    my_string_data._cfisa,
    my_string_data._cfinfo[0],my_string_data._cfinfo[1],my_string_data._cfinfo[2],my_string_data._cfinfo[3],
    my_string_data._rc,
    my_string_data.variants.inline1.length,
    my_string_data.variants.notInlineImmutable1.buffer,
    my_string_data.variants.notInlineImmutable1.length,
    my_string_data.variants.notInlineImmutable1.contentsDeallocator,
    my_string_data.variants.notInlineImmutable2.buffer,
    my_string_data.variants.notInlineImmutable2.contentsDeallocator,
    my_string_data.variants.notInlineMutable.buffer,
    my_string_data.variants.notInlineMutable.length,
    my_string_data.variants.notInlineMutable.capacity,
    my_string_data.variants.notInlineMutable.hasGap,
    my_string_data.variants.notInlineMutable.isFixedCapacity,
    my_string_data.variants.notInlineMutable.isExternalMutable,
    my_string_data.variants.notInlineMutable.capacityProvidedExternally,
    my_string_data.variants.notInlineMutable.desiredCapacity,
    my_string_data.variants.notInlineMutable.desiredCapacity,
    my_string_data.variants.notInlineMutable.contentsAllocator,
    is_mutable,
    is_inline,
    explicit_length,
    is_unicode,
    is_special,
    has_null);
#endif
    
    if (strcmp(class_name,"NSString") &&
        strcmp(class_name,"CFStringRef") &&
        strcmp(class_name,"CFMutableStringRef") &&
        strcmp(class_name,"__NSCFConstantString") &&
        strcmp(class_name,"__NSCFString") &&
        strcmp(class_name,"NSCFConstantString") &&
        strcmp(class_name,"NSCFString") &&
        strcmp(class_name,"NSPathStore2"))
    {
        // not one of us - but tell me class name
        stream.Printf("class name = %s",class_name);
        return true;
    }
    
    if (is_mutable)
    {
        uint64_t location = 2 * ptr_size + valobj_addr;
        location = process_sp->ReadPointerFromMemory(location, error);
        if (error.Fail())
            return false;
        if (has_explicit_length and is_unicode)
        {
            ReadUTFBufferAndDumpToStreamOptions<UTF16> options;
            options.SetConversionFunction(ConvertUTF16toUTF8);
            options.SetLocation(location);
            options.SetProcessSP(process_sp);
            options.SetStream(&stream);
            options.SetPrefixToken('@');
            options.SetQuote('"');
            options.SetSourceSize(explicit_length);
            options.SetNeedsZeroTermination(false);
            return ReadUTFBufferAndDumpToStream (options);
        }
        else
            return ReadAsciiBufferAndDumpToStream(location+1,process_sp,stream, explicit_length);
    }
    else if (is_inline && has_explicit_length && !is_unicode && !is_special && !is_mutable)
    {
        uint64_t location = 3 * ptr_size + valobj_addr;
        return ReadAsciiBufferAndDumpToStream(location,process_sp,stream,explicit_length);
    }
    else if (is_unicode)
    {
        uint64_t location = valobj_addr + 2*ptr_size;
        if (is_inline)
        {
            if (!has_explicit_length)
            {
                stream.Printf("found new combo");
                return true;
            }
            else
                location += ptr_size;
        }
        else
        {
            location = process_sp->ReadPointerFromMemory(location, error);
            if (error.Fail())
                return false;
        }
        ReadUTFBufferAndDumpToStreamOptions<UTF16> options;
        options.SetConversionFunction(ConvertUTF16toUTF8);
        options.SetLocation(location);
        options.SetProcessSP(process_sp);
        options.SetStream(&stream);
        options.SetPrefixToken('@');
        options.SetQuote('"');
        options.SetSourceSize(explicit_length);
        options.SetNeedsZeroTermination(has_explicit_length == false);
        return ReadUTFBufferAndDumpToStream (options);
    }
    else if (is_special)
    {
        uint64_t location = valobj_addr + (ptr_size == 8 ? 12 : 8);
        ReadUTFBufferAndDumpToStreamOptions<UTF16> options;
        options.SetConversionFunction(ConvertUTF16toUTF8);
        options.SetLocation(location);
        options.SetProcessSP(process_sp);
        options.SetStream(&stream);
        options.SetPrefixToken('@');
        options.SetQuote('"');
        options.SetSourceSize(explicit_length);
        options.SetNeedsZeroTermination(has_explicit_length == false);
        return ReadUTFBufferAndDumpToStream (options);
    }
    else if (is_inline)
    {
        uint64_t location = valobj_addr + 2*ptr_size;
        if (!has_explicit_length)
            location++;
        return ReadAsciiBufferAndDumpToStream(location,process_sp,stream,explicit_length);
    }
    else
    {
        uint64_t location = valobj_addr + 2*ptr_size;
        location = process_sp->ReadPointerFromMemory(location, error);
        if (error.Fail())
            return false;
        return ReadAsciiBufferAndDumpToStream(location,process_sp,stream,explicit_length);
    }
    
    stream.Printf("class name = %s",class_name);
    return true;
    
}

bool
lldb_private::formatters::NSAttributedStringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    TargetSP target_sp(valobj.GetTargetSP());
    if (!target_sp)
        return false;
    uint32_t addr_size = target_sp->GetArchitecture().GetAddressByteSize();
    uint64_t pointee = valobj.GetValueAsUnsigned(0);
    if (!pointee)
        return false;
    pointee += addr_size;
    ClangASTType type(valobj.GetClangAST(),valobj.GetClangType());
    ExecutionContext exe_ctx(target_sp,false);
    ValueObjectSP child_ptr_sp(valobj.CreateValueObjectFromAddress("string_ptr", pointee, exe_ctx, type));
    if (!child_ptr_sp)
        return false;
    DataExtractor data;
    child_ptr_sp->GetData(data);
    ValueObjectSP child_sp(child_ptr_sp->CreateValueObjectFromData("string_data", data, exe_ctx, type));
    child_sp->GetValueAsUnsigned(0);
    if (child_sp)
        return NSStringSummaryProvider(*child_sp, stream);
    return false;
}

bool
lldb_private::formatters::NSMutableAttributedStringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    return NSAttributedStringSummaryProvider(valobj, stream);
}

bool
lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider (ValueObject& valobj, Stream& stream)
{
    stream.Printf("%s",valobj.GetObjectDescription());
    return true;
}

bool
lldb_private::formatters::ObjCBOOLSummaryProvider (ValueObject& valobj, Stream& stream)
{
    const uint32_t type_info = ClangASTContext::GetTypeInfo(valobj.GetClangType(),
                                                            valobj.GetClangAST(),
                                                            NULL);
    
    ValueObjectSP real_guy_sp = valobj.GetSP();
    
    if (type_info & ClangASTContext::eTypeIsPointer)
    {
        Error err;
        real_guy_sp = valobj.Dereference(err);
        if (err.Fail() || !real_guy_sp)
            return false;
    }
    else if (type_info & ClangASTContext::eTypeIsReference)
    {
        real_guy_sp =  valobj.GetChildAtIndex(0, true);
        if (!real_guy_sp)
            return false;
    }
    uint64_t value = real_guy_sp->GetValueAsUnsigned(0);
    if (value == 0)
    {
        stream.Printf("NO");
        return true;
    }
    stream.Printf("YES");
    return true;
}

template <bool is_sel_ptr>
bool
lldb_private::formatters::ObjCSELSummaryProvider (ValueObject& valobj, Stream& stream)
{
    lldb::ValueObjectSP valobj_sp;

    if (!valobj.GetClangAST())
        return false;
    void* char_opaque_type = valobj.GetClangAST()->CharTy.getAsOpaquePtr();
    if (!char_opaque_type)
        return false;
    ClangASTType charstar(valobj.GetClangAST(),ClangASTType::GetPointerType(valobj.GetClangAST(), char_opaque_type));

    ExecutionContext exe_ctx(valobj.GetExecutionContextRef());
    
    if (is_sel_ptr)
    {
        lldb::addr_t data_address = valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
        if (data_address == LLDB_INVALID_ADDRESS)
            return false;
        valobj_sp = ValueObject::CreateValueObjectFromAddress("text", data_address, exe_ctx, charstar);
    }
    else
    {
        DataExtractor data;
        valobj.GetData(data);
        valobj_sp = ValueObject::CreateValueObjectFromData("text", data, exe_ctx, charstar);
    }
    
    if (!valobj_sp)
        return false;
    
    stream.Printf("%s",valobj_sp->GetSummaryAsCString());
    return true;
}

// POSIX has an epoch on Jan-1-1970, but Cocoa prefers Jan-1-2001
// this call gives the POSIX equivalent of the Cocoa epoch
time_t
lldb_private::formatters::GetOSXEpoch ()
{
    static time_t epoch = 0;
    if (!epoch)
    {
        tzset();
        tm tm_epoch;
        tm_epoch.tm_sec = 0;
        tm_epoch.tm_hour = 0;
        tm_epoch.tm_min = 0;
        tm_epoch.tm_mon = 0;
        tm_epoch.tm_mday = 1;
        tm_epoch.tm_year = 2001-1900; // for some reason, we need to subtract 1900 from this field. not sure why.
        tm_epoch.tm_isdst = -1;
        tm_epoch.tm_gmtoff = 0;
        tm_epoch.tm_zone = NULL;
        epoch = timegm(&tm_epoch);
    }
    return epoch;
}

size_t
lldb_private::formatters::ExtractIndexFromString (const char* item_name)
{
    if (!item_name || !*item_name)
        return UINT32_MAX;
    if (*item_name != '[')
        return UINT32_MAX;
    item_name++;
    char* endptr = NULL;
    unsigned long int idx = ::strtoul(item_name, &endptr, 0);
    if (idx == 0 && endptr == item_name)
        return UINT32_MAX;
    if (idx == ULONG_MAX)
        return UINT32_MAX;
    return idx;
}

lldb_private::formatters::VectorIteratorSyntheticFrontEnd::VectorIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp,
                                                                                            ConstString item_name) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_item_name(item_name),
m_item_sp()
{
    if (valobj_sp)
        Update();
}

bool
lldb_private::formatters::VectorIteratorSyntheticFrontEnd::Update()
{
    m_item_sp.reset();

    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    
    if (!valobj_sp)
        return false;
    
    ValueObjectSP item_ptr(valobj_sp->GetChildMemberWithName(m_item_name,true));
    if (!item_ptr)
        return false;
    if (item_ptr->GetValueAsUnsigned(0) == 0)
        return false;
    Error err;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    m_item_sp = ValueObject::CreateValueObjectFromAddress("item", item_ptr->GetValueAsUnsigned(0), m_exe_ctx_ref, ClangASTType(item_ptr->GetClangAST(),ClangASTType::GetPointeeType(item_ptr->GetClangType())));
    if (err.Fail())
        m_item_sp.reset();
    return false;
}

size_t
lldb_private::formatters::VectorIteratorSyntheticFrontEnd::CalculateNumChildren ()
{
    return 1;
}

lldb::ValueObjectSP
lldb_private::formatters::VectorIteratorSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (idx == 0)
        return m_item_sp;
    return lldb::ValueObjectSP();
}

bool
lldb_private::formatters::VectorIteratorSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::VectorIteratorSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (name == ConstString("item"))
        return 0;
    return UINT32_MAX;
}

lldb_private::formatters::VectorIteratorSyntheticFrontEnd::~VectorIteratorSyntheticFrontEnd ()
{
}

template bool
lldb_private::formatters::NSDataSummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::NSDataSummaryProvider<false> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::ObjCSELSummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::ObjCSELSummaryProvider<false> (ValueObject&, Stream&) ;
