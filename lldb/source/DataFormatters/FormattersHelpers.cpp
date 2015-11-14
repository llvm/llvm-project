//===-- FormattersHelpers.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes

// C++ Includes

// Other libraries and framework includes

// Project includes
#include "lldb/DataFormatters/FormattersHelpers.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

void
lldb_private::formatters::AddFormat (TypeCategoryImpl::SharedPointer category_sp,
                                     lldb::Format format,
                                     ConstString type_name,
                                     TypeFormatImpl::Flags flags,
                                     bool regex)
{
    lldb::TypeFormatImplSP format_sp(new TypeFormatImpl_Format(format, flags));
    
    if (regex)
        category_sp->GetRegexTypeFormatsContainer()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())),format_sp);
    else
        category_sp->GetTypeFormatsContainer()->Add(type_name, format_sp);
}

void
lldb_private::formatters::AddSummary(TypeCategoryImpl::SharedPointer category_sp,
                                     TypeSummaryImplSP summary_sp,
                                     ConstString type_name,
                                     bool regex)
{
    if (regex)
        category_sp->GetRegexTypeSummariesContainer()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())),summary_sp);
    else
        category_sp->GetTypeSummariesContainer()->Add(type_name, summary_sp);
}

void
lldb_private::formatters::AddStringSummary(TypeCategoryImpl::SharedPointer category_sp,
                                           const char* string,
                                           ConstString type_name,
                                           TypeSummaryImpl::Flags flags,
                                           bool regex)
{
    lldb::TypeSummaryImplSP summary_sp(new StringSummaryFormat(flags,
                                                               string));
    
    if (regex)
        category_sp->GetRegexTypeSummariesContainer()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())),summary_sp);
    else
        category_sp->GetTypeSummariesContainer()->Add(type_name, summary_sp);
}

void
lldb_private::formatters::AddOneLineSummary (TypeCategoryImpl::SharedPointer category_sp,
                                             ConstString type_name,
                                             TypeSummaryImpl::Flags flags,
                                             bool regex)
{
    flags.SetShowMembersOneLiner(true);
    lldb::TypeSummaryImplSP summary_sp(new StringSummaryFormat(flags, ""));
    
    if (regex)
        category_sp->GetRegexTypeSummariesContainer()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())),summary_sp);
    else
        category_sp->GetTypeSummariesContainer()->Add(type_name, summary_sp);
}

#ifndef LLDB_DISABLE_PYTHON
void
lldb_private::formatters::AddCXXSummary (TypeCategoryImpl::SharedPointer category_sp,
                                         CXXFunctionSummaryFormat::Callback funct,
                                         const char* description,
                                         ConstString type_name,
                                         TypeSummaryImpl::Flags flags,
                                         bool regex)
{
    lldb::TypeSummaryImplSP summary_sp(new CXXFunctionSummaryFormat(flags,funct,description));
    if (regex)
        category_sp->GetRegexTypeSummariesContainer()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())),summary_sp);
    else
        category_sp->GetTypeSummariesContainer()->Add(type_name, summary_sp);
}

void
lldb_private::formatters::AddCXXSynthetic  (TypeCategoryImpl::SharedPointer category_sp,
                                            CXXSyntheticChildren::CreateFrontEndCallback generator,
                                            const char* description,
                                            ConstString type_name,
                                            ScriptedSyntheticChildren::Flags flags,
                                            bool regex)
{
    lldb::SyntheticChildrenSP synth_sp(new CXXSyntheticChildren(flags,description,generator));
    if (regex)
        category_sp->GetRegexTypeSyntheticsContainer()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())), synth_sp);
    else
        category_sp->GetTypeSyntheticsContainer()->Add(type_name,synth_sp);
}

void
lldb_private::formatters::AddFilter  (TypeCategoryImpl::SharedPointer category_sp,
                                      std::vector<std::string> children,
                                      const char* description,
                                      ConstString type_name,
                                      ScriptedSyntheticChildren::Flags flags,
                                      bool regex)
{
    TypeFilterImplSP filter_sp(new TypeFilterImpl(flags));
    for (auto child : children)
        filter_sp->AddExpressionPath(child);
    if (regex)
        category_sp->GetRegexTypeFiltersContainer()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())), filter_sp);
    else
        category_sp->GetTypeFiltersContainer()->Add(type_name,filter_sp);
}
#endif

StackFrame*
lldb_private::formatters::GetViableFrame (ExecutionContext exe_ctx)
{
    StackFrame* frame = exe_ctx.GetFramePtr();
    if (frame)
        return frame;
    
    Process* process = exe_ctx.GetProcessPtr();
    if (!process)
        return nullptr;
    
    ThreadSP thread_sp(process->GetThreadList().GetSelectedThread());
    if (thread_sp)
        return thread_sp->GetSelectedFrame().get();
    return nullptr;
}

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
    StackFrame* stack_frame = GetViableFrame(exe_ctx);
    if (!target || !stack_frame)
        return false;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false);
    options.SetUnwindOnError(true);
    options.SetKeepInMemory(true);
    options.SetLanguage(lldb::eLanguageTypeObjC_plus_plus);
    options.SetResultIsInternal(true);
    options.SetUseDynamic(lldb::eDynamicCanRunTarget);

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
                                                            Stream &stream,
                                                            lldb::LanguageType lang_type)
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
    StackFrame* stack_frame = GetViableFrame(exe_ctx);
    if (!target || !stack_frame)
        return false;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false);
    options.SetUnwindOnError(true);
    options.SetKeepInMemory(true);
    options.SetLanguage(lldb::eLanguageTypeObjC_plus_plus);
    options.SetResultIsInternal(true);
    options.SetUseDynamic(lldb::eDynamicCanRunTarget);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               result_sp,
                               options);
    if (!result_sp)
        return false;
    stream.Printf("%s",result_sp->GetSummaryAsCString(lang_type));
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
    StreamString expr;
    const char *colon = "";
    llvm::StringRef selector_sr(selector);
    if (selector_sr.back() != ':')
        colon = ":";
    expr.Printf("(%s)[(id)0x%" PRIx64 " %s%s%" PRId64 "]",return_type,valobj.GetPointerValue(),selector,colon,index);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = GetViableFrame(exe_ctx);
    if (!target || !stack_frame)
        return valobj_sp;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false);
    options.SetUnwindOnError(true);
    options.SetKeepInMemory(true);
    options.SetLanguage(lldb::eLanguageTypeObjC_plus_plus);
    options.SetResultIsInternal(true);
    options.SetUseDynamic(lldb::eDynamicCanRunTarget);
    
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
    StreamString expr;
    const char *colon = "";
    llvm::StringRef selector_sr(selector);
    if (selector_sr.back() != ':')
        colon = ":";
    expr.Printf("(%s)[(id)0x%" PRIx64 " %s%s%s]",return_type,valobj.GetPointerValue(),selector,colon,key);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = GetViableFrame(exe_ctx);
    if (!target || !stack_frame)
        return valobj_sp;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false);
    options.SetUnwindOnError(true);
    options.SetKeepInMemory(true);
    options.SetLanguage(lldb::eLanguageTypeObjC_plus_plus);
    options.SetResultIsInternal(true);
    options.SetUseDynamic(lldb::eDynamicCanRunTarget);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               valobj_sp,
                               options);
    return valobj_sp;
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

lldb::addr_t
lldb_private::formatters::GetArrayAddressOrPointerValue (ValueObject& valobj)
{
    lldb::addr_t data_addr = LLDB_INVALID_ADDRESS;

    if (valobj.IsPointerType())
        data_addr = valobj.GetValueAsUnsigned(0);
    else if (valobj.IsArrayType())
        data_addr = valobj.GetAddressOf();

    return data_addr;
}
