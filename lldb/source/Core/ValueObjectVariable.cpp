//===-- ValueObjectVariable.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/Core/ValueObjectVariable.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Module.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/Value.h"

#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolContextScope.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"


using namespace lldb_private;

lldb::ValueObjectSP
ValueObjectVariable::Create (ExecutionContextScope *exe_scope, const lldb::VariableSP &var_sp)
{
    return (new ValueObjectVariable (exe_scope, var_sp))->GetSP();
}

ValueObjectVariable::ValueObjectVariable (ExecutionContextScope *exe_scope, const lldb::VariableSP &var_sp) :
    ValueObject(exe_scope),
    m_variable_sp(var_sp)
{
    // Do not attempt to construct one of these objects with no variable!
    assert (m_variable_sp.get() != NULL);
    m_name = var_sp->GetName();
}

ValueObjectVariable::~ValueObjectVariable()
{
}

lldb::clang_type_t
ValueObjectVariable::GetClangTypeImpl ()
{
    Type *var_type = m_variable_sp->GetType();
    if (var_type)
        return var_type->GetClangForwardType();
    return NULL;
}

ConstString
ValueObjectVariable::GetTypeName()
{
    Type * var_type = m_variable_sp->GetType();
    if (var_type)
        return var_type->GetName();
    return ConstString();
}

ConstString
ValueObjectVariable::GetQualifiedTypeName()
{
    Type * var_type = m_variable_sp->GetType();
    if (var_type)
        return var_type->GetQualifiedName();
    return ConstString();
}

uint32_t
ValueObjectVariable::CalculateNumChildren()
{    
    ClangASTType type(GetClangAST(),
                      GetClangType());
    
    if (!type.IsValid())
        return 0;
    
    const bool omit_empty_base_classes = true;
    return ClangASTContext::GetNumChildren(type.GetASTContext(), type.GetOpaqueQualType(), omit_empty_base_classes);
}

clang::ASTContext *
ValueObjectVariable::GetClangASTImpl ()
{
    Type *var_type = m_variable_sp->GetType();
    if (var_type)
        return var_type->GetClangAST();
    return 0;
}

size_t
ValueObjectVariable::GetByteSize()
{
    ClangASTType type(GetClangAST(),
                      GetClangType());
    
    if (!type.IsValid())
        return 0;
    
    return (ClangASTType::GetClangTypeBitWidth(type.GetASTContext(), type.GetOpaqueQualType()) + 7) / 8;
}

lldb::ValueType
ValueObjectVariable::GetValueType() const
{
    if (m_variable_sp)
        return m_variable_sp->GetScope();
    return lldb::eValueTypeInvalid;
}

bool
ValueObjectVariable::UpdateValue ()
{
    SetValueIsValid (false);
    m_error.Clear();

    Variable *variable = m_variable_sp.get();
    DWARFExpression &expr = variable->LocationExpression();
    
    if (variable->GetLocationIsConstantValueData())
    {
        // expr doesn't contain DWARF bytes, it contains the constant variable
        // value bytes themselves...
        if (expr.GetExpressionData(m_data))
            m_value.SetContext(Value::eContextTypeVariable, variable);
        else
            m_error.SetErrorString ("empty constant data");
    }
    else
    {
        lldb::addr_t loclist_base_load_addr = LLDB_INVALID_ADDRESS;
        ExecutionContext exe_ctx (GetExecutionContextRef());
        
        Target *target = exe_ctx.GetTargetPtr();
        if (target)
        {
            m_data.SetByteOrder(target->GetArchitecture().GetByteOrder());
            m_data.SetAddressByteSize(target->GetArchitecture().GetAddressByteSize());
        }

        if (expr.IsLocationList())
        {
            SymbolContext sc;
            variable->CalculateSymbolContext (&sc);
            if (sc.function)
                loclist_base_load_addr = sc.function->GetAddressRange().GetBaseAddress().GetLoadAddress (target);
        }
        Value old_value(m_value);
        if (expr.Evaluate (&exe_ctx, GetClangAST(), NULL, NULL, NULL, loclist_base_load_addr, NULL, m_value, &m_error))
        {
            m_value.SetContext(Value::eContextTypeVariable, variable);

            Value::ValueType value_type = m_value.GetValueType();
            
            switch (value_type)
            {
                case Value::eValueTypeFileAddress:
                    SetAddressTypeOfChildren(eAddressTypeFile);
                    break;
                case Value::eValueTypeHostAddress:
                    SetAddressTypeOfChildren(eAddressTypeHost);
                    break;
                case Value::eValueTypeLoadAddress:
                case Value::eValueTypeScalar:
                case Value::eValueTypeVector:
                    SetAddressTypeOfChildren(eAddressTypeLoad);
                    break;
            }

            switch (value_type)
            {
            default:
                assert(!"Unhandled expression result value kind...");
                break;

            case Value::eValueTypeScalar:
                // The variable value is in the Scalar value inside the m_value.
                // We can point our m_data right to it.
                m_error = m_value.GetValueAsData (&exe_ctx, GetClangAST(), m_data, 0, GetModule().get());
                break;

            case Value::eValueTypeFileAddress:
            case Value::eValueTypeLoadAddress:
            case Value::eValueTypeHostAddress:
                // The DWARF expression result was an address in the inferior
                // process. If this variable is an aggregate type, we just need
                // the address as the main value as all child variable objects
                // will rely upon this location and add an offset and then read
                // their own values as needed. If this variable is a simple
                // type, we read all data for it into m_data.
                // Make sure this type has a value before we try and read it

                // If we have a file address, convert it to a load address if we can.
                Process *process = exe_ctx.GetProcessPtr();
                if (value_type == Value::eValueTypeFileAddress && process && process->IsAlive())
                {
                    lldb::addr_t file_addr = m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
                    if (file_addr != LLDB_INVALID_ADDRESS)
                    {
                        SymbolContext var_sc;
                        variable->CalculateSymbolContext(&var_sc);
                        if (var_sc.module_sp)
                        {
                            ObjectFile *objfile = var_sc.module_sp->GetObjectFile();
                            if (objfile)
                            {
                                Address so_addr(file_addr, objfile->GetSectionList());
                                lldb::addr_t load_addr = so_addr.GetLoadAddress (target);
                                if (load_addr != LLDB_INVALID_ADDRESS)
                                {
                                    m_value.SetValueType(Value::eValueTypeLoadAddress);
                                    m_value.GetScalar() = load_addr;
                                }
                            }
                        }
                    }
                }

                if (ClangASTContext::IsAggregateType (GetClangType()))
                {
                    // this value object represents an aggregate type whose
                    // children have values, but this object does not. So we
                    // say we are changed if our location has changed.
                    SetValueDidChange (value_type != old_value.GetValueType() || m_value.GetScalar() != old_value.GetScalar());
                }
                else
                {
                    // Copy the Value and set the context to use our Variable
                    // so it can extract read its value into m_data appropriately
                    Value value(m_value);
                    value.SetContext(Value::eContextTypeVariable, variable);
                    m_error = value.GetValueAsData(&exe_ctx, GetClangAST(), m_data, 0, GetModule().get());
                }
                break;
            }

            SetValueIsValid (m_error.Success());
        }
    }
    return m_error.Success();
}



bool
ValueObjectVariable::IsInScope ()
{
    const ExecutionContextRef &exe_ctx_ref = GetExecutionContextRef();
    if (exe_ctx_ref.HasFrameRef())
    {
        ExecutionContext exe_ctx (exe_ctx_ref);
        StackFrame *frame = exe_ctx.GetFramePtr();
        if (frame)
        {
            return m_variable_sp->IsInScope (frame);
        }
        else
        {
            // This ValueObject had a frame at one time, but now we
            // can't locate it, so return false since we probably aren't
            // in scope.
            return false;
        }
    }
    // We have a variable that wasn't tied to a frame, which
    // means it is a global and is always in scope.
    return true;
         
}

lldb::ModuleSP
ValueObjectVariable::GetModule()
{
    if (m_variable_sp)
    {
        SymbolContextScope *sc_scope = m_variable_sp->GetSymbolContextScope();
        if (sc_scope)
        {
            return sc_scope->CalculateSymbolContextModule();
        }
    }
    return lldb::ModuleSP();
}

SymbolContextScope *
ValueObjectVariable::GetSymbolContextScope()
{
    if (m_variable_sp)
        return m_variable_sp->GetSymbolContextScope();
    return NULL;
}

bool
ValueObjectVariable::GetDeclaration (Declaration &decl)
{
    if (m_variable_sp)
    {
        decl = m_variable_sp->GetDeclaration();
        return true;
    }
    return false;
}
