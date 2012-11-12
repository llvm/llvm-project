//===-- SBValue.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBValue.h"

#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBTypeFilter.h"
#include "lldb/API/SBTypeFormat.h"
#include "lldb/API/SBTypeSummary.h"
#include "lldb/API/SBTypeSynthetic.h"

#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/DataVisualization.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Declaration.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBExpressionOptions.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"

using namespace lldb;
using namespace lldb_private;

namespace {
    class ValueImpl
    {
    public:
        ValueImpl ()
        {
        }
        
        ValueImpl (lldb::ValueObjectSP opaque_sp,
                   lldb::DynamicValueType use_dynamic,
                   bool use_synthetic) :
            m_opaque_sp(opaque_sp),
            m_use_dynamic(use_dynamic),
            m_use_synthetic(use_synthetic)
        {
        }
        
        ValueImpl (const ValueImpl& rhs) :
            m_opaque_sp(rhs.m_opaque_sp),
            m_use_dynamic(rhs.m_use_dynamic),
            m_use_synthetic(rhs.m_use_synthetic)
        {
        }
        
        ValueImpl &
        operator = (const ValueImpl &rhs)
        {
            if (this != &rhs)
            {
                m_opaque_sp = rhs.m_opaque_sp;
                m_use_dynamic = rhs.m_use_dynamic;
                m_use_synthetic = rhs.m_use_synthetic;
            }
            return *this;
        }
        
        bool
        IsValid ()
        {
            return m_opaque_sp.get() != NULL;
        }
        
        lldb::ValueObjectSP
        GetRootSP ()
        {
            return m_opaque_sp;
        }
        
        lldb::ValueObjectSP
        GetSP ()
        {
            if (!m_opaque_sp)
                return m_opaque_sp;
            lldb::ValueObjectSP value_sp = m_opaque_sp;
            if (value_sp->GetDynamicValue(m_use_dynamic))
                value_sp = value_sp->GetDynamicValue(m_use_dynamic);
            if (value_sp->GetSyntheticValue(m_use_synthetic))
                value_sp = value_sp->GetSyntheticValue(m_use_synthetic);
            return value_sp;
        }
        
        void
        SetUseDynamic (lldb::DynamicValueType use_dynamic)
        {
            m_use_dynamic = use_dynamic;
        }
        
        void
        SetUseSynthetic (bool use_synthetic)
        {
            m_use_synthetic = use_synthetic;
        }
        
        lldb::DynamicValueType
        GetUseDynamic ()
        {
            return m_use_dynamic;
        }
        
        bool
        GetUseSynthetic ()
        {
            return m_use_synthetic;
        }
        
    private:
        lldb::ValueObjectSP m_opaque_sp;
        lldb::DynamicValueType m_use_dynamic;
        bool m_use_synthetic;
    };
}

SBValue::SBValue () :
    m_opaque_sp ()
{
}

SBValue::SBValue (const lldb::ValueObjectSP &value_sp)
{
    SetSP(value_sp);
}

SBValue::SBValue(const SBValue &rhs)
{
    SetSP(rhs.m_opaque_sp);
}

SBValue &
SBValue::operator = (const SBValue &rhs)
{
    if (this != &rhs)
    {
        SetSP(rhs.m_opaque_sp);
    }
    return *this;
}

SBValue::~SBValue()
{
}

bool
SBValue::IsValid ()
{
    // If this function ever changes to anything that does more than just
    // check if the opaque shared pointer is non NULL, then we need to update
    // all "if (m_opaque_sp)" code in this file.
    return m_opaque_sp.get() != NULL && m_opaque_sp->GetRootSP().get() != NULL;
}

void
SBValue::Clear()
{
    m_opaque_sp.reset();
}

SBError
SBValue::GetError()
{
    SBError sb_error;
    
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        sb_error.SetError(value_sp->GetError());
    else
        sb_error.SetErrorString("error: invalid value");
    
    return sb_error;
}

user_id_t
SBValue::GetID()
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        return value_sp->GetID();
    return LLDB_INVALID_UID;
}

const char *
SBValue::GetName()
{

    const char *name = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        name = value_sp->GetName().GetCString();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (name)
            log->Printf ("SBValue(%p)::GetName () => \"%s\"", value_sp.get(), name);
        else
            log->Printf ("SBValue(%p)::GetName () => NULL", value_sp.get());
    }

    return name;
}

const char *
SBValue::GetTypeName ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    const char *name = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        // For a dynamic type we might have to run code to determine the type we are going to report,
        // and we might not have updated the type before we get asked this.  So make sure to get the API lock.
        
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetTypeName() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                name = value_sp->GetQualifiedTypeName().GetCString();
            }
        }
    }
    
    if (log)
    {
        if (name)
            log->Printf ("SBValue(%p)::GetTypeName () => \"%s\"", value_sp.get(), name);
        else
            log->Printf ("SBValue(%p)::GetTypeName () => NULL", value_sp.get());
    }

    return name;
}

size_t
SBValue::GetByteSize ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    size_t result = 0;

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        // For a dynamic type we might have to run code to determine the type we are going to report,
        // and we might not have updated the type before we get asked this.  So make sure to get the API lock.
        
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetTypeName() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                result = value_sp->GetByteSize();
            }
        }
    }

    if (log)
        log->Printf ("SBValue(%p)::GetByteSize () => %llu", value_sp.get(), (uint64_t)result);

    return result;
}

bool
SBValue::IsInScope ()
{
    bool result = false;

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
            result = value_sp->IsInScope ();
        }
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::IsInScope () => %i", value_sp.get(), result);

    return result;
}

const char *
SBValue::GetValue ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    const char *cstr = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetValue() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                cstr = value_sp->GetValueAsCString ();
            }
        }
    }
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetValue() => \"%s\"", value_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetValue() => NULL", value_sp.get());
    }

    return cstr;
}

ValueType
SBValue::GetValueType ()
{
    ValueType result = eValueTypeInvalid;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        result = value_sp->GetValueType();
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        switch (result)
        {
        case eValueTypeInvalid:         log->Printf ("SBValue(%p)::GetValueType () => eValueTypeInvalid", value_sp.get()); break;
        case eValueTypeVariableGlobal:  log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableGlobal", value_sp.get()); break;
        case eValueTypeVariableStatic:  log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableStatic", value_sp.get()); break;
        case eValueTypeVariableArgument:log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableArgument", value_sp.get()); break;
        case eValueTypeVariableLocal:   log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableLocal", value_sp.get()); break;
        case eValueTypeRegister:        log->Printf ("SBValue(%p)::GetValueType () => eValueTypeRegister", value_sp.get()); break;
        case eValueTypeRegisterSet:     log->Printf ("SBValue(%p)::GetValueType () => eValueTypeRegisterSet", value_sp.get()); break;
        case eValueTypeConstResult:     log->Printf ("SBValue(%p)::GetValueType () => eValueTypeConstResult", value_sp.get()); break;
        default:     log->Printf ("SBValue(%p)::GetValueType () => %i ???", value_sp.get(), result); break;
        }
    }
    return result;
}

const char *
SBValue::GetObjectDescription ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    const char *cstr = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetObjectDescription() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                cstr = value_sp->GetObjectDescription ();
            }
        }
    }
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetObjectDescription() => \"%s\"", value_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetObjectDescription() => NULL", value_sp.get());
    }
    return cstr;
}

SBType
SBValue::GetType()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    SBType sb_type;
    lldb::ValueObjectSP value_sp(GetSP());
    TypeImplSP type_sp;
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetType() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                type_sp.reset (new TypeImpl(ClangASTType (value_sp->GetClangAST(), value_sp->GetClangType())));
                sb_type.SetSP(type_sp);
            }
        }
    }
    if (log)
    {
        if (type_sp)
            log->Printf ("SBValue(%p)::GetType => SBType(%p)", value_sp.get(), type_sp.get());
        else
            log->Printf ("SBValue(%p)::GetType => NULL", value_sp.get());
    }
    return sb_type;
}

bool
SBValue::GetValueDidChange ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    bool result = false;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetValueDidChange() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                result = value_sp->GetValueDidChange ();
            }
        }
    }
    if (log)
        log->Printf ("SBValue(%p)::GetValueDidChange() => %i", value_sp.get(), result);

    return result;
}

#ifndef LLDB_DISABLE_PYTHON
const char *
SBValue::GetSummary ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    const char *cstr = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetSummary() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                cstr = value_sp->GetSummaryAsCString();
            }
        }
    }
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetSummary() => \"%s\"", value_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetSummary() => NULL", value_sp.get());
    }
    return cstr;
}
#endif // LLDB_DISABLE_PYTHON

const char *
SBValue::GetLocation ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    const char *cstr = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetLocation() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                cstr = value_sp->GetLocationAsCString();
            }
        }
    }
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetLocation() => \"%s\"", value_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetLocation() => NULL", value_sp.get());
    }
    return cstr;
}

// Deprecated - use the one that takes an lldb::SBError
bool
SBValue::SetValueFromCString (const char *value_str)
{
    lldb::SBError dummy;
    return SetValueFromCString(value_str,dummy);
}

bool
SBValue::SetValueFromCString (const char *value_str, lldb::SBError& error)
{
    bool success = false;
    lldb::ValueObjectSP value_sp(GetSP());
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::SetValueFromCString() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                success = value_sp->SetValueFromCString (value_str,error.ref());
            }
        }
    }
    if (log)
        log->Printf ("SBValue(%p)::SetValueFromCString(\"%s\") => %i", value_sp.get(), value_str, success);

    return success;
}

lldb::SBTypeFormat
SBValue::GetTypeFormat ()
{
    lldb::SBTypeFormat format;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::GetTypeFormat() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                if (value_sp->UpdateValueIfNeeded(true))
                {
                    lldb::TypeFormatImplSP format_sp = value_sp->GetValueFormat();
                    if (format_sp)
                        format.SetSP(format_sp);
                }
            }
        }
    }
    return format;
}

#ifndef LLDB_DISABLE_PYTHON
lldb::SBTypeSummary
SBValue::GetTypeSummary ()
{
    lldb::SBTypeSummary summary;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::GetTypeSummary() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                if (value_sp->UpdateValueIfNeeded(true))
                {
                    lldb::TypeSummaryImplSP summary_sp = value_sp->GetSummaryFormat();
                    if (summary_sp)
                        summary.SetSP(summary_sp);
                }
            }
        }
    }
    return summary;
}
#endif // LLDB_DISABLE_PYTHON

lldb::SBTypeFilter
SBValue::GetTypeFilter ()
{
    lldb::SBTypeFilter filter;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::GetTypeFilter() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                if (value_sp->UpdateValueIfNeeded(true))
                {
                    lldb::SyntheticChildrenSP synthetic_sp = value_sp->GetSyntheticChildren();
                    
                    if (synthetic_sp && !synthetic_sp->IsScripted())
                    {
                        TypeFilterImplSP filter_sp = STD_STATIC_POINTER_CAST(TypeFilterImpl,synthetic_sp);
                        filter.SetSP(filter_sp);
                    }
                }
            }
        }
    }
    return filter;
}

#ifndef LLDB_DISABLE_PYTHON
lldb::SBTypeSynthetic
SBValue::GetTypeSynthetic ()
{
    lldb::SBTypeSynthetic synthetic;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::GetTypeSynthetic() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                if (value_sp->UpdateValueIfNeeded(true))
                {
                    lldb::SyntheticChildrenSP children_sp = value_sp->GetSyntheticChildren();
                    
                    if (children_sp && children_sp->IsScripted())
                    {
                        TypeSyntheticImplSP synth_sp = STD_STATIC_POINTER_CAST(TypeSyntheticImpl,children_sp);
                        synthetic.SetSP(synth_sp);
                    }
                }
            }
        }
    }
    return synthetic;
}
#endif

lldb::SBValue
SBValue::CreateChildAtOffset (const char *name, uint32_t offset, SBType type)
{
    lldb::SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    lldb::ValueObjectSP new_value_sp;
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::CreateChildAtOffset() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                TypeImplSP type_sp (type.GetSP());
                if (type.IsValid())
                {
                    sb_value.SetSP(value_sp->GetSyntheticChildAtOffset(offset, type_sp->GetClangASTType(), true),GetPreferDynamicValue(),GetPreferSyntheticValue());
                    new_value_sp = sb_value.GetSP();
                    if (new_value_sp)
                        new_value_sp->SetName(ConstString(name));
                }
            }
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (new_value_sp)
            log->Printf ("SBValue(%p)::CreateChildAtOffset => \"%s\"",
                         value_sp.get(),
                         new_value_sp->GetName().AsCString());
        else
            log->Printf ("SBValue(%p)::CreateChildAtOffset => NULL",
                         value_sp.get());
    }
    return sb_value;
}

lldb::SBValue
SBValue::Cast (SBType type)
{
    lldb::SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    TypeImplSP type_sp (type.GetSP());
    if (value_sp && type_sp)
        sb_value.SetSP(value_sp->Cast(type_sp->GetClangASTType()),GetPreferDynamicValue(),GetPreferSyntheticValue());
    return sb_value;
}

lldb::SBValue
SBValue::CreateValueFromExpression (const char *name, const char* expression)
{
    SBExpressionOptions options;
    options.ref().SetKeepInMemory(true);
    return CreateValueFromExpression (name, expression, options);
}

lldb::SBValue
SBValue::CreateValueFromExpression (const char *name, const char *expression, SBExpressionOptions &options)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    lldb::SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    lldb::ValueObjectSP new_value_sp;
    if (value_sp)
    {
        ExecutionContext exe_ctx (value_sp->GetExecutionContextRef());
        ProcessSP process_sp(exe_ctx.GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::CreateValueFromExpression() => error: process is running", value_sp.get());
        }
        else
        {
            Target* target = exe_ctx.GetTargetPtr();
            if (target)
            {
                options.ref().SetKeepInMemory(true);
                target->EvaluateExpression (expression,
                                            exe_ctx.GetFramePtr(),
                                            new_value_sp,
                                            options.ref());
                if (new_value_sp)
                {
                    new_value_sp->SetName(ConstString(name));
                    sb_value.SetSP(new_value_sp);
                }
            }
        }
    }
    if (log)
    {
        if (new_value_sp)
            log->Printf ("SBValue(%p)::CreateValueFromExpression(name=\"%s\", expression=\"%s\") => SBValue (%p)",
                         value_sp.get(),
                         name,
                         expression,
                         new_value_sp.get());
        else
            log->Printf ("SBValue(%p)::CreateValueFromExpression(name=\"%s\", expression=\"%s\") => NULL",
                         value_sp.get(),
                         name,
                         expression);
    }
    return sb_value;
}

lldb::SBValue
SBValue::CreateValueFromAddress(const char* name, lldb::addr_t address, SBType sb_type)
{
    lldb::SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    lldb::ValueObjectSP new_value_sp;
    lldb::TypeImplSP type_impl_sp (sb_type.GetSP());
    if (value_sp && type_impl_sp)
    {
        ClangASTType pointee_ast_type(type_impl_sp->GetASTContext(), type_impl_sp->GetClangASTType().GetPointerType ());
        lldb::TypeImplSP pointee_type_impl_sp (new TypeImpl(pointee_ast_type));
        if (pointee_type_impl_sp)
        {
        
            lldb::DataBufferSP buffer(new lldb_private::DataBufferHeap(&address,sizeof(lldb::addr_t)));
        
            ExecutionContext exe_ctx (value_sp->GetExecutionContextRef());
            ValueObjectSP ptr_result_valobj_sp(ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(),
                                                                               pointee_type_impl_sp->GetASTContext(),
                                                                               pointee_type_impl_sp->GetOpaqueQualType(),
                                                                               ConstString(name),
                                                                               buffer,
                                                                               lldb::endian::InlHostByteOrder(), 
                                                                               exe_ctx.GetAddressByteSize()));
        
            if (ptr_result_valobj_sp)
            {
                ptr_result_valobj_sp->GetValue().SetValueType(Value::eValueTypeLoadAddress);
                Error err;
                new_value_sp = ptr_result_valobj_sp->Dereference(err);
                if (new_value_sp)
                    new_value_sp->SetName(ConstString(name));
            }
            sb_value.SetSP(new_value_sp);
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (new_value_sp)
            log->Printf ("SBValue(%p)::CreateValueFromAddress => \"%s\"", value_sp.get(), new_value_sp->GetName().AsCString());
        else
            log->Printf ("SBValue(%p)::CreateValueFromAddress => NULL", value_sp.get());
    }
    return sb_value;
}

lldb::SBValue
SBValue::CreateValueFromData (const char* name, SBData data, SBType type)
{
    lldb::SBValue sb_value;
    lldb::ValueObjectSP new_value_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ExecutionContext exe_ctx (value_sp->GetExecutionContextRef());

        new_value_sp = ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(),
                                                       type.m_opaque_sp->GetASTContext() ,
                                                       type.m_opaque_sp->GetOpaqueQualType(),
                                                       ConstString(name),
                                                       *data.m_opaque_sp,
                                                       LLDB_INVALID_ADDRESS);
        new_value_sp->SetAddressTypeOfChildren(eAddressTypeLoad);
        sb_value.SetSP(new_value_sp);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (new_value_sp)
            log->Printf ("SBValue(%p)::CreateValueFromData => \"%s\"", value_sp.get(), new_value_sp->GetName().AsCString());
        else
            log->Printf ("SBValue(%p)::CreateValueFromData => NULL", value_sp.get());
    }
    return sb_value;
}

SBValue
SBValue::GetChildAtIndex (uint32_t idx)
{
    const bool can_create_synthetic = false;
    lldb::DynamicValueType use_dynamic = eNoDynamicValues;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetTargetSP());
        if (target_sp)
            use_dynamic = target_sp->GetPreferDynamicValue();
    }
    return GetChildAtIndex (idx, use_dynamic, can_create_synthetic);
}

SBValue
SBValue::GetChildAtIndex (uint32_t idx, lldb::DynamicValueType use_dynamic, bool can_create_synthetic)
{
    lldb::ValueObjectSP child_sp;
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetChildAtIndex() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                const bool can_create = true;
                child_sp = value_sp->GetChildAtIndex (idx, can_create);
                if (can_create_synthetic && !child_sp)
                {
                    if (value_sp->IsPointerType())
                    {
                        child_sp = value_sp->GetSyntheticArrayMemberFromPointer(idx, can_create);
                    }
                    else if (value_sp->IsArrayType())
                    {
                        child_sp = value_sp->GetSyntheticArrayMemberFromArray(idx, can_create);
                    }
                }
                    
            }
        }
    }
    
    SBValue sb_value;
    sb_value.SetSP (child_sp, use_dynamic, GetPreferSyntheticValue());
    if (log)
        log->Printf ("SBValue(%p)::GetChildAtIndex (%u) => SBValue(%p)", value_sp.get(), idx, value_sp.get());

    return sb_value;
}

uint32_t
SBValue::GetIndexOfChildWithName (const char *name)
{
    uint32_t idx = UINT32_MAX;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
        
            idx = value_sp->GetIndexOfChildWithName (ConstString(name));
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (idx == UINT32_MAX)
            log->Printf ("SBValue(%p)::GetIndexOfChildWithName (name=\"%s\") => NOT FOUND", value_sp.get(), name);
        else
            log->Printf ("SBValue(%p)::GetIndexOfChildWithName (name=\"%s\") => %u", value_sp.get(), name, idx);
    }
    return idx;
}

SBValue
SBValue::GetChildMemberWithName (const char *name)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        lldb::DynamicValueType use_dynamic_value = eNoDynamicValues;
        TargetSP target_sp(value_sp->GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
            use_dynamic_value = target_sp->GetPreferDynamicValue();
        }
        return GetChildMemberWithName (name, use_dynamic_value);
    }
    return SBValue();
}

SBValue
SBValue::GetChildMemberWithName (const char *name, lldb::DynamicValueType use_dynamic_value)
{
    lldb::ValueObjectSP child_sp;
    const ConstString str_name (name);

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetChildMemberWithName() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                child_sp = value_sp->GetChildMemberWithName (str_name, true);
            }
        }
    }
    
    SBValue sb_value;
    sb_value.SetSP(child_sp, use_dynamic_value, GetPreferSyntheticValue());

    if (log)
        log->Printf ("SBValue(%p)::GetChildMemberWithName (name=\"%s\") => SBValue(%p)", value_sp.get(), name, value_sp.get());

    return sb_value;
}

lldb::SBValue
SBValue::GetDynamicValue (lldb::DynamicValueType use_dynamic)
{
    SBValue value_sb;
    if (IsValid())
    {
        ValueImplSP proxy_sp(new ValueImpl(m_opaque_sp->GetRootSP(),use_dynamic,m_opaque_sp->GetUseSynthetic()));
        value_sb.SetSP(proxy_sp);
    }
    return value_sb;
}

lldb::SBValue
SBValue::GetStaticValue ()
{
    SBValue value_sb;
    if (IsValid())
    {
        ValueImplSP proxy_sp(new ValueImpl(m_opaque_sp->GetRootSP(),eNoDynamicValues,m_opaque_sp->GetUseSynthetic()));
        value_sb.SetSP(proxy_sp);
    }
    return value_sb;
}

lldb::SBValue
SBValue::GetNonSyntheticValue ()
{
    SBValue value_sb;
    if (IsValid())
    {
        ValueImplSP proxy_sp(new ValueImpl(m_opaque_sp->GetRootSP(),m_opaque_sp->GetUseDynamic(),false));
        value_sb.SetSP(proxy_sp);
    }
    return value_sb;
}

lldb::DynamicValueType
SBValue::GetPreferDynamicValue ()
{
    if (!IsValid())
        return eNoDynamicValues;
    return m_opaque_sp->GetUseDynamic();
}

void
SBValue::SetPreferDynamicValue (lldb::DynamicValueType use_dynamic)
{
    if (IsValid())
        return m_opaque_sp->SetUseDynamic (use_dynamic);
}

bool
SBValue::GetPreferSyntheticValue ()
{
    if (!IsValid())
        return false;
    return m_opaque_sp->GetUseSynthetic();
}

void
SBValue::SetPreferSyntheticValue (bool use_synthetic)
{
    if (IsValid())
        return m_opaque_sp->SetUseSynthetic (use_synthetic);
}

bool
SBValue::IsDynamic()
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
            return value_sp->IsDynamic();
        }
    }
    return false;
}

bool
SBValue::IsSynthetic ()
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
            return value_sp->IsSynthetic();
        }
    }
    return false;
}

lldb::SBValue
SBValue::GetValueForExpressionPath(const char* expr_path)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    lldb::ValueObjectSP child_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetValueForExpressionPath() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                // using default values for all the fancy options, just do it if you can
                child_sp = value_sp->GetValueForExpressionPath(expr_path);
            }
        }
    }
    
    SBValue sb_value;
    sb_value.SetSP(child_sp,GetPreferDynamicValue(),GetPreferSyntheticValue());
    
    if (log)
        log->Printf ("SBValue(%p)::GetValueForExpressionPath (expr_path=\"%s\") => SBValue(%p)", value_sp.get(), expr_path, value_sp.get());
    
    return sb_value;
}

int64_t
SBValue::GetValueAsSigned(SBError& error, int64_t fail_value)
{
    error.Clear();
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::GetValueAsSigned() => error: process is running", value_sp.get());
            error.SetErrorString("process is running");
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                Scalar scalar;
                if (value_sp->ResolveValue (scalar))
                    return scalar.SLongLong(fail_value);
                else
                    error.SetErrorString("could not get value");
            }
            else
                error.SetErrorString("could not get target");
        }
    }
    error.SetErrorString("invalid SBValue");
    return fail_value;
}

uint64_t
SBValue::GetValueAsUnsigned(SBError& error, uint64_t fail_value)
{
    error.Clear();
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::GetValueAsUnsigned() => error: process is running", value_sp.get());
            error.SetErrorString("process is running");
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                Scalar scalar;
                if (value_sp->ResolveValue (scalar))
                    return scalar.ULongLong(fail_value);
                else
                    error.SetErrorString("could not get value");
            }
            else
                error.SetErrorString("could not get target");
        }
    }
    error.SetErrorString("invalid SBValue");
    return fail_value;
}

int64_t
SBValue::GetValueAsSigned(int64_t fail_value)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::GetValueAsSigned() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                Scalar scalar;
                if (value_sp->ResolveValue (scalar))
                    return scalar.SLongLong(fail_value);
            }
        }
    }
    return fail_value;
}

uint64_t
SBValue::GetValueAsUnsigned(uint64_t fail_value)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::GetValueAsUnsigned() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                Scalar scalar;
                if (value_sp->ResolveValue (scalar))
                    return scalar.ULongLong(fail_value);
            }
        }
    }
    return fail_value;
}

bool
SBValue::MightHaveChildren ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    bool has_children = false;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        has_children = value_sp->MightHaveChildren();

    if (log)
        log->Printf ("SBValue(%p)::HasChildren() => %i", value_sp.get(), has_children);
    return has_children;
}

uint32_t
SBValue::GetNumChildren ()
{
    uint32_t num_children = 0;

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetNumChildren() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp(value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());

                num_children = value_sp->GetNumChildren();
            }
        }
    }

    if (log)
        log->Printf ("SBValue(%p)::GetNumChildren () => %u", value_sp.get(), num_children);

    return num_children;
}


SBValue
SBValue::Dereference ()
{
    SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());

            Error error;
            sb_value = value_sp->Dereference (error);
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::Dereference () => SBValue(%p)", value_sp.get(), value_sp.get());

    return sb_value;
}

bool
SBValue::TypeIsPointerType ()
{
    bool is_ptr_type = false;

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());

            is_ptr_type = value_sp->IsPointerType();
        }
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::TypeIsPointerType () => %i", value_sp.get(), is_ptr_type);


    return is_ptr_type;
}

void *
SBValue::GetOpaqueType()
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());

            return value_sp->GetClangType();
        }
    }
    return NULL;
}

lldb::SBTarget
SBValue::GetTarget()
{
    SBTarget sb_target;
    TargetSP target_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        target_sp = value_sp->GetTargetSP();
        sb_target.SetSP (target_sp);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (target_sp.get() == NULL)
            log->Printf ("SBValue(%p)::GetTarget () => NULL", value_sp.get());
        else
            log->Printf ("SBValue(%p)::GetTarget () => %p", value_sp.get(), target_sp.get());
    }
    return sb_target;
}

lldb::SBProcess
SBValue::GetProcess()
{
    SBProcess sb_process;
    ProcessSP process_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        process_sp = value_sp->GetProcessSP();
        if (process_sp)
            sb_process.SetSP (process_sp);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (process_sp.get() == NULL)
            log->Printf ("SBValue(%p)::GetProcess () => NULL", value_sp.get());
        else
            log->Printf ("SBValue(%p)::GetProcess () => %p", value_sp.get(), process_sp.get());
    }
    return sb_process;
}

lldb::SBThread
SBValue::GetThread()
{
    SBThread sb_thread;
    ThreadSP thread_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        thread_sp = value_sp->GetThreadSP();
        sb_thread.SetThread(thread_sp);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (thread_sp.get() == NULL)
            log->Printf ("SBValue(%p)::GetThread () => NULL", value_sp.get());
        else
            log->Printf ("SBValue(%p)::GetThread () => %p", value_sp.get(), thread_sp.get());
    }
    return sb_thread;
}

lldb::SBFrame
SBValue::GetFrame()
{
    SBFrame sb_frame;
    StackFrameSP frame_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        frame_sp = value_sp->GetFrameSP();
        sb_frame.SetFrameSP (frame_sp);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (frame_sp.get() == NULL)
            log->Printf ("SBValue(%p)::GetFrame () => NULL", value_sp.get());
        else
            log->Printf ("SBValue(%p)::GetFrame () => %p", value_sp.get(), frame_sp.get());
    }
    return sb_frame;
}


lldb::ValueObjectSP
SBValue::GetSP () const
{
    if (!m_opaque_sp || !m_opaque_sp->IsValid())
        return ValueObjectSP();
    return m_opaque_sp->GetSP();
}

void
SBValue::SetSP (ValueImplSP impl_sp)
{
    m_opaque_sp = impl_sp;
}

void
SBValue::SetSP (const lldb::ValueObjectSP &sp)
{
    if (sp)
    {
        lldb::TargetSP target_sp(sp->GetTargetSP());
        if (target_sp)
        {
            lldb::DynamicValueType use_dynamic = target_sp->GetPreferDynamicValue();
            bool use_synthetic = target_sp->TargetProperties::GetEnableSyntheticValue();
            m_opaque_sp = ValueImplSP(new ValueImpl(sp, use_dynamic, use_synthetic));
        }
        else
            m_opaque_sp = ValueImplSP(new ValueImpl(sp,eNoDynamicValues,true));
    }
    else
        m_opaque_sp = ValueImplSP(new ValueImpl(sp,eNoDynamicValues,false));
}

void
SBValue::SetSP (const lldb::ValueObjectSP &sp, lldb::DynamicValueType use_dynamic)
{
    if (sp)
    {
        lldb::TargetSP target_sp(sp->GetTargetSP());
        if (target_sp)
        {
            bool use_synthetic = target_sp->TargetProperties::GetEnableSyntheticValue();
            SetSP (sp, use_dynamic, use_synthetic);
        }
        else
            SetSP (sp, use_dynamic, true);
    }
    else
        SetSP (sp, use_dynamic, false);
}

void
SBValue::SetSP (const lldb::ValueObjectSP &sp, bool use_synthetic)
{
    if (sp)
    {
        lldb::TargetSP target_sp(sp->GetTargetSP());
        if (target_sp)
        {
            lldb::DynamicValueType use_dynamic = target_sp->GetPreferDynamicValue();
            SetSP (sp, use_dynamic, use_synthetic);
        }
        else
            SetSP (sp, eNoDynamicValues, use_synthetic);
    }
    else
        SetSP (sp, eNoDynamicValues, use_synthetic);
}

void
SBValue::SetSP (const lldb::ValueObjectSP &sp, lldb::DynamicValueType use_dynamic, bool use_synthetic)
{
    m_opaque_sp = ValueImplSP(new ValueImpl(sp,use_dynamic,use_synthetic));
}

bool
SBValue::GetExpressionPath (SBStream &description)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        value_sp->GetExpressionPath (description.ref(), false);
        return true;
    }
    return false;
}

bool
SBValue::GetExpressionPath (SBStream &description, bool qualify_cxx_base_classes)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        value_sp->GetExpressionPath (description.ref(), qualify_cxx_base_classes);
        return true;
    }
    return false;
}

bool
SBValue::GetDescription (SBStream &description)
{
    Stream &strm = description.ref();

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::GetDescription() => error: process is running", value_sp.get());
        }
        else
        {
            ValueObject::DumpValueObject (strm, value_sp.get());
        }
    }
    else
        strm.PutCString ("No value");

    return true;
}

lldb::Format
SBValue::GetFormat ()
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        return value_sp->GetFormat();
    return eFormatDefault;
}

void
SBValue::SetFormat (lldb::Format format)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        value_sp->SetFormat(format);
}

lldb::SBValue
SBValue::AddressOf()
{
    SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp (value_sp->GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
            Error error;
            sb_value.SetSP(value_sp->AddressOf (error),GetPreferDynamicValue(), GetPreferSyntheticValue());
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::AddressOf () => SBValue(%p)", value_sp.get(), value_sp.get());
    
    return sb_value;
}

lldb::addr_t
SBValue::GetLoadAddress()
{
    lldb::addr_t value = LLDB_INVALID_ADDRESS;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp (value_sp->GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
            const bool scalar_is_load_address = true;
            AddressType addr_type;
            value = value_sp->GetAddressOf(scalar_is_load_address, &addr_type);
            if (addr_type == eAddressTypeFile)
            {
                ModuleSP module_sp (value_sp->GetModule());
                if (!module_sp)
                    value = LLDB_INVALID_ADDRESS;
                else
                {
                    Address addr;
                    module_sp->ResolveFileAddress(value, addr);
                    value = addr.GetLoadAddress(target_sp.get());
                }
            }
            else if (addr_type == eAddressTypeHost || addr_type == eAddressTypeInvalid)
                value = LLDB_INVALID_ADDRESS;
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetLoadAddress () => (%llu)", value_sp.get(), value);
    
    return value;
}

lldb::SBAddress
SBValue::GetAddress()
{
    Address addr;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp (value_sp->GetTargetSP());
        if (target_sp)
        {
            lldb::addr_t value = LLDB_INVALID_ADDRESS;
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
            const bool scalar_is_load_address = true;
            AddressType addr_type;
            value = value_sp->GetAddressOf(scalar_is_load_address, &addr_type);
            if (addr_type == eAddressTypeFile)
            {
                ModuleSP module_sp (value_sp->GetModule());
                if (module_sp)
                    module_sp->ResolveFileAddress(value, addr);
            }
            else if (addr_type == eAddressTypeLoad)
            {
                // no need to check the return value on this.. if it can actually do the resolve
                // addr will be in the form (section,offset), otherwise it will simply be returned
                // as (NULL, value)
                addr.SetLoadAddress(value, target_sp.get());
            }
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetAddress () => (%s,%llu)", value_sp.get(),
                     (addr.GetSection() ? addr.GetSection()->GetName().GetCString() : "NULL"),
                     addr.GetOffset());
    return SBAddress(new Address(addr));
}

lldb::SBData
SBValue::GetPointeeData (uint32_t item_idx,
                         uint32_t item_count)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    lldb::SBData sb_data;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetPointeeData() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp (value_sp->GetTargetSP());
            if (target_sp)
            {
                DataExtractorSP data_sp(new DataExtractor());
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                value_sp->GetPointeeData(*data_sp, item_idx, item_count);
                if (data_sp->GetByteSize() > 0)
                    *sb_data = data_sp;
            }
        }
    }
    if (log)
        log->Printf ("SBValue(%p)::GetPointeeData (%d, %d) => SBData(%p)",
                     value_sp.get(),
                     item_idx,
                     item_count,
                     sb_data.get());
    
    return sb_data;
}

lldb::SBData
SBValue::GetData ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    lldb::SBData sb_data;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            if (log)
                log->Printf ("SBValue(%p)::GetData() => error: process is running", value_sp.get());
        }
        else
        {
            TargetSP target_sp (value_sp->GetTargetSP());
            if (target_sp)
            {
                Mutex::Locker api_locker (target_sp->GetAPIMutex());
                DataExtractorSP data_sp(new DataExtractor());
                value_sp->GetData(*data_sp);
                if (data_sp->GetByteSize() > 0)
                    *sb_data = data_sp;
            }
        }
    }
    if (log)
        log->Printf ("SBValue(%p)::GetData () => SBData(%p)",
                     value_sp.get(),
                     sb_data.get());
    
    return sb_data;
}

lldb::SBDeclaration
SBValue::GetDeclaration ()
{
    lldb::ValueObjectSP value_sp(GetSP());
    SBDeclaration decl_sb;
    if (value_sp)
    {
        Declaration decl;
        if (value_sp->GetDeclaration(decl))
            decl_sb.SetDeclaration(decl);
    }
    return decl_sb;
}

lldb::SBWatchpoint
SBValue::Watch (bool resolve_location, bool read, bool write, SBError &error)
{
    SBWatchpoint sb_watchpoint;
    
    // If the SBValue is not valid, there's no point in even trying to watch it.
    lldb::ValueObjectSP value_sp(GetSP());
    TargetSP target_sp (GetTarget().GetSP());
    if (value_sp && target_sp)
    {
        // Can't watch this if the process is running
        ProcessSP process_sp(value_sp->GetProcessSP());
        Process::StopLocker stop_locker;
        if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBValue(%p)::Watch() => error: process is running", value_sp.get());
            return sb_watchpoint;
        }

        // Read and Write cannot both be false.
        if (!read && !write)
            return sb_watchpoint;
        
        // If the value is not in scope, don't try and watch and invalid value
        if (!IsInScope())
            return sb_watchpoint;
        
        addr_t addr = GetLoadAddress();
        if (addr == LLDB_INVALID_ADDRESS)
            return sb_watchpoint;
        size_t byte_size = GetByteSize();
        if (byte_size == 0)
            return sb_watchpoint;
                
        uint32_t watch_type = 0;
        if (read)
            watch_type |= LLDB_WATCH_TYPE_READ;
        if (write)
            watch_type |= LLDB_WATCH_TYPE_WRITE;
        
        Error rc;
        ClangASTType type (value_sp->GetClangAST(), value_sp->GetClangType());
        WatchpointSP watchpoint_sp = target_sp->CreateWatchpoint(addr, byte_size, &type, watch_type, rc);
        error.SetError(rc);
                
        if (watchpoint_sp) 
        {
            sb_watchpoint.SetSP (watchpoint_sp);
            Declaration decl;
            if (value_sp->GetDeclaration (decl))
            {
                if (decl.GetFile()) 
                {
                    StreamString ss;
                    // True to show fullpath for declaration file.
                    decl.DumpStopContext(&ss, true);
                    watchpoint_sp->SetDeclInfo(ss.GetString());
                }
            }
        }
    }
    return sb_watchpoint;
}

// FIXME: Remove this method impl (as well as the decl in .h) once it is no longer needed.
// Backward compatibility fix in the interim.
lldb::SBWatchpoint
SBValue::Watch (bool resolve_location, bool read, bool write)
{
    SBError error;
    return Watch(resolve_location, read, write, error);
}

lldb::SBWatchpoint
SBValue::WatchPointee (bool resolve_location, bool read, bool write, SBError &error)
{
    SBWatchpoint sb_watchpoint;
    if (IsInScope() && GetType().IsPointerType())
        sb_watchpoint = Dereference().Watch (resolve_location, read, write, error);
    return sb_watchpoint;
}
