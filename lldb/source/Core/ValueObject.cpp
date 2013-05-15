//===-- ValueObject.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Core/ValueObject.h"

// C Includes
#include <stdlib.h>

// C++ Includes
// Other libraries and framework includes
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/Type.h"

// Project includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectCast.h"
#include "lldb/Core/ValueObjectChild.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectDynamicValue.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Core/ValueObjectSyntheticFilter.h"

#include "lldb/DataFormatters/DataVisualization.h"

#include "lldb/Host/Endian.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_utility;

static user_id_t g_value_obj_uid = 0;

//----------------------------------------------------------------------
// ValueObject constructor
//----------------------------------------------------------------------
ValueObject::ValueObject (ValueObject &parent) :
    UserID (++g_value_obj_uid), // Unique identifier for every value object
    m_parent (&parent),
    m_root (NULL),
    m_update_point (parent.GetUpdatePoint ()),
    m_name (),
    m_data (),
    m_value (),
    m_error (),
    m_value_str (),
    m_old_value_str (),
    m_location_str (),
    m_summary_str (),
    m_object_desc_str (),
    m_manager(parent.GetManager()),
    m_children (),
    m_synthetic_children (),
    m_dynamic_value (NULL),
    m_synthetic_value(NULL),
    m_deref_valobj(NULL),
    m_format (eFormatDefault),
    m_last_format_mgr_revision(0),
    m_type_summary_sp(),
    m_type_format_sp(),
    m_synthetic_children_sp(),
    m_user_id_of_forced_summary(),
    m_address_type_of_ptr_or_ref_children(eAddressTypeInvalid),
    m_value_is_valid (false),
    m_value_did_change (false),
    m_children_count_valid (false),
    m_old_value_valid (false),
    m_is_deref_of_parent (false),
    m_is_array_item_for_pointer(false),
    m_is_bitfield_for_scalar(false),
    m_is_child_at_offset(false),
    m_is_getting_summary(false),
    m_did_calculate_complete_objc_class_type(false)
{
    m_manager->ManageObject(this);
}

//----------------------------------------------------------------------
// ValueObject constructor
//----------------------------------------------------------------------
ValueObject::ValueObject (ExecutionContextScope *exe_scope,
                          AddressType child_ptr_or_ref_addr_type) :
    UserID (++g_value_obj_uid), // Unique identifier for every value object
    m_parent (NULL),
    m_root (NULL),
    m_update_point (exe_scope),
    m_name (),
    m_data (),
    m_value (),
    m_error (),
    m_value_str (),
    m_old_value_str (),
    m_location_str (),
    m_summary_str (),
    m_object_desc_str (),
    m_manager(),
    m_children (),
    m_synthetic_children (),
    m_dynamic_value (NULL),
    m_synthetic_value(NULL),
    m_deref_valobj(NULL),
    m_format (eFormatDefault),
    m_last_format_mgr_revision(0),
    m_type_summary_sp(),
    m_type_format_sp(),
    m_synthetic_children_sp(),
    m_user_id_of_forced_summary(),
    m_address_type_of_ptr_or_ref_children(child_ptr_or_ref_addr_type),
    m_value_is_valid (false),
    m_value_did_change (false),
    m_children_count_valid (false),
    m_old_value_valid (false),
    m_is_deref_of_parent (false),
    m_is_array_item_for_pointer(false),
    m_is_bitfield_for_scalar(false),
    m_is_child_at_offset(false),
    m_is_getting_summary(false),
    m_did_calculate_complete_objc_class_type(false)
{
    m_manager = new ValueObjectManager();
    m_manager->ManageObject (this);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ValueObject::~ValueObject ()
{
}

bool
ValueObject::UpdateValueIfNeeded (bool update_format)
{
    
    bool did_change_formats = false;
    
    if (update_format)
        did_change_formats = UpdateFormatsIfNeeded();
    
    // If this is a constant value, then our success is predicated on whether
    // we have an error or not
    if (GetIsConstant())
    {
        // if you were asked to update your formatters, but did not get a chance to do it
        // clear your own values (this serves the purpose of faking a stop-id for frozen
        // objects (which are regarded as constant, but could have changes behind their backs
        // because of the frozen-pointer depth limit)
		// TODO: decouple summary from value and then remove this code and only force-clear the summary
        if (update_format && !did_change_formats)
            ClearUserVisibleData(eClearUserVisibleDataItemsSummary);
        return m_error.Success();
    }

    bool first_update = m_update_point.IsFirstEvaluation();
    
    if (m_update_point.NeedsUpdating())
    {
        m_update_point.SetUpdated();
        
        // Save the old value using swap to avoid a string copy which
        // also will clear our m_value_str
        if (m_value_str.empty())
        {
            m_old_value_valid = false;
        }
        else
        {
            m_old_value_valid = true;
            m_old_value_str.swap (m_value_str);
            ClearUserVisibleData(eClearUserVisibleDataItemsValue);
        }

        ClearUserVisibleData();
        
        if (IsInScope())
        {
            const bool value_was_valid = GetValueIsValid();
            SetValueDidChange (false);
            
            m_error.Clear();

            // Call the pure virtual function to update the value
            bool success = UpdateValue ();
            
            SetValueIsValid (success);
            
            if (first_update)
                SetValueDidChange (false);
            else if (!m_value_did_change && success == false)
            {
                // The value wasn't gotten successfully, so we mark this
                // as changed if the value used to be valid and now isn't
                SetValueDidChange (value_was_valid);
            }
        }
        else
        {
            m_error.SetErrorString("out of scope");
        }
    }
    return m_error.Success();
}

bool
ValueObject::UpdateFormatsIfNeeded()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    if (log)
        log->Printf("[%s %p] checking for FormatManager revisions. ValueObject rev: %d - Global rev: %d",
           GetName().GetCString(),
           this,
           m_last_format_mgr_revision,
           DataVisualization::GetCurrentRevision());
    
    bool any_change = false;
    
    if ( (m_last_format_mgr_revision != DataVisualization::GetCurrentRevision()))
    {
        SetValueFormat(DataVisualization::ValueFormats::GetFormat (*this, eNoDynamicValues));
        SetSummaryFormat(DataVisualization::GetSummaryFormat (*this, GetDynamicValueType()));
#ifndef LLDB_DISABLE_PYTHON
        SetSyntheticChildren(DataVisualization::GetSyntheticChildren (*this, GetDynamicValueType()));
#endif

        m_last_format_mgr_revision = DataVisualization::GetCurrentRevision();
        
        any_change = true;
    }
    
    return any_change;
    
}

void
ValueObject::SetNeedsUpdate ()
{
    m_update_point.SetNeedsUpdate();
    // We have to clear the value string here so ConstResult children will notice if their values are
    // changed by hand (i.e. with SetValueAsCString).
    ClearUserVisibleData(eClearUserVisibleDataItemsValue);
}

void
ValueObject::ClearDynamicTypeInformation ()
{
    m_did_calculate_complete_objc_class_type = false;
    m_last_format_mgr_revision = 0;
    m_override_type = ClangASTType();
    SetValueFormat(lldb::TypeFormatImplSP());
    SetSummaryFormat(lldb::TypeSummaryImplSP());
    SetSyntheticChildren(lldb::SyntheticChildrenSP());
}

ClangASTType
ValueObject::MaybeCalculateCompleteType ()
{
    ClangASTType ret(GetClangASTImpl(), GetClangTypeImpl());
        
    if (m_did_calculate_complete_objc_class_type)
    {
        if (m_override_type.IsValid())
            return m_override_type;
        else
            return ret;
    }
    
    clang_type_t ast_type(GetClangTypeImpl());
    clang_type_t class_type;
    bool is_pointer_type;
    
    if (ClangASTContext::IsObjCObjectPointerType(ast_type, &class_type))
    {
        is_pointer_type = true;
    }
    else if (ClangASTContext::IsObjCClassType(ast_type))
    {
        is_pointer_type = false;
        class_type = ast_type;
    }
    else
    {
        return ret;
    }
    
    m_did_calculate_complete_objc_class_type = true;
    
    if (!class_type)
        return ret;
    
    std::string class_name;
    
    if (!ClangASTContext::GetObjCClassName(class_type, class_name))
        return ret;
    
    ProcessSP process_sp(GetUpdatePoint().GetExecutionContextRef().GetProcessSP());
    
    if (!process_sp)
        return ret;
    
    ObjCLanguageRuntime *objc_language_runtime(process_sp->GetObjCLanguageRuntime());
    
    if (!objc_language_runtime)
        return ret;
    
    ConstString class_name_cs(class_name.c_str());
    
    TypeSP complete_objc_class_type_sp = objc_language_runtime->LookupInCompleteClassCache(class_name_cs);
    
    if (!complete_objc_class_type_sp)
        return ret;
    
    ClangASTType complete_class(complete_objc_class_type_sp->GetClangAST(),
                                complete_objc_class_type_sp->GetClangFullType());
    
    if (!ClangASTContext::GetCompleteType(complete_class.GetASTContext(), 
                                          complete_class.GetOpaqueQualType()))
        return ret;
    
    if (is_pointer_type)
    {
        clang_type_t pointer_type = ClangASTContext::CreatePointerType(complete_class.GetASTContext(),
                                                                       complete_class.GetOpaqueQualType());
        
        m_override_type = ClangASTType(complete_class.GetASTContext(),
                                       pointer_type);
    }
    else
    {
        m_override_type = complete_class;
    }
    
    if (m_override_type.IsValid())
        return m_override_type;
    else
        return ret;
}

clang::ASTContext *
ValueObject::GetClangAST ()
{
    ClangASTType type = MaybeCalculateCompleteType();
    
    return type.GetASTContext();
}

lldb::clang_type_t
ValueObject::GetClangType ()
{
    ClangASTType type = MaybeCalculateCompleteType();
    
    return type.GetOpaqueQualType();
}

DataExtractor &
ValueObject::GetDataExtractor ()
{
    UpdateValueIfNeeded(false);
    return m_data;
}

const Error &
ValueObject::GetError()
{
    UpdateValueIfNeeded(false);
    return m_error;
}

const ConstString &
ValueObject::GetName() const
{
    return m_name;
}

const char *
ValueObject::GetLocationAsCString ()
{
    return GetLocationAsCStringImpl(m_value,
                                    m_data);
}

const char *
ValueObject::GetLocationAsCStringImpl (const Value& value,
                                       const DataExtractor& data)
{
    if (UpdateValueIfNeeded(false))
    {
        if (m_location_str.empty())
        {
            StreamString sstr;
            
            Value::ValueType value_type = value.GetValueType();
            
            switch (value_type)
            {
            case Value::eValueTypeScalar:
            case Value::eValueTypeVector:
                if (value.GetContextType() == Value::eContextTypeRegisterInfo)
                {
                    RegisterInfo *reg_info = value.GetRegisterInfo();
                    if (reg_info)
                    {
                        if (reg_info->name)
                            m_location_str = reg_info->name;
                        else if (reg_info->alt_name)
                            m_location_str = reg_info->alt_name;
                        if (m_location_str.empty())
                            m_location_str = (reg_info->encoding == lldb::eEncodingVector) ? "vector" : "scalar";
                    }
                }
                if (m_location_str.empty())
                    m_location_str = (value_type == Value::eValueTypeVector) ? "vector" : "scalar";
                break;

            case Value::eValueTypeLoadAddress:
            case Value::eValueTypeFileAddress:
            case Value::eValueTypeHostAddress:
                {
                    uint32_t addr_nibble_size = data.GetAddressByteSize() * 2;
                    sstr.Printf("0x%*.*llx", addr_nibble_size, addr_nibble_size, value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS));
                    m_location_str.swap(sstr.GetString());
                }
                break;
            }
        }
    }
    return m_location_str.c_str();
}

Value &
ValueObject::GetValue()
{
    return m_value;
}

const Value &
ValueObject::GetValue() const
{
    return m_value;
}

bool
ValueObject::ResolveValue (Scalar &scalar)
{
    if (UpdateValueIfNeeded(false)) // make sure that you are up to date before returning anything
    {
        ExecutionContext exe_ctx (GetExecutionContextRef());
        Value tmp_value(m_value);
        scalar = tmp_value.ResolveValue(&exe_ctx, GetClangAST ());
        if (scalar.IsValid())
        {
            const uint32_t bitfield_bit_size = GetBitfieldBitSize();
            if (bitfield_bit_size)
                return scalar.ExtractBitfield (bitfield_bit_size, GetBitfieldBitOffset());
            return true;
        }
    }
    return false;
}

bool
ValueObject::GetValueIsValid () const
{
    return m_value_is_valid;
}


void
ValueObject::SetValueIsValid (bool b)
{
    m_value_is_valid = b;
}

bool
ValueObject::GetValueDidChange ()
{
    GetValueAsCString ();
    return m_value_did_change;
}

void
ValueObject::SetValueDidChange (bool value_changed)
{
    m_value_did_change = value_changed;
}

ValueObjectSP
ValueObject::GetChildAtIndex (size_t idx, bool can_create)
{
    ValueObjectSP child_sp;
    // We may need to update our value if we are dynamic
    if (IsPossibleDynamicType ())
        UpdateValueIfNeeded(false);
    if (idx < GetNumChildren())
    {
        // Check if we have already made the child value object?
        if (can_create && !m_children.HasChildAtIndex(idx))
        {
            // No we haven't created the child at this index, so lets have our
            // subclass do it and cache the result for quick future access.
            m_children.SetChildAtIndex(idx,CreateChildAtIndex (idx, false, 0));
        }
        
        ValueObject* child = m_children.GetChildAtIndex(idx);
        if (child != NULL)
            return child->GetSP();
    }
    return child_sp;
}

ValueObjectSP
ValueObject::GetChildAtIndexPath (const std::initializer_list<size_t>& idxs,
                                  size_t* index_of_error)
{
    if (idxs.size() == 0)
        return GetSP();
    ValueObjectSP root(GetSP());
    for (size_t idx : idxs)
    {
        root = root->GetChildAtIndex(idx, true);
        if (!root)
        {
            if (index_of_error)
                *index_of_error = idx;
            return root;
        }
    }
    return root;
}

ValueObjectSP
ValueObject::GetChildAtIndexPath (const std::initializer_list< std::pair<size_t, bool> >& idxs,
                                  size_t* index_of_error)
{
    if (idxs.size() == 0)
        return GetSP();
    ValueObjectSP root(GetSP());
    for (std::pair<size_t, bool> idx : idxs)
    {
        root = root->GetChildAtIndex(idx.first, idx.second);
        if (!root)
        {
            if (index_of_error)
                *index_of_error = idx.first;
            return root;
        }
    }
    return root;
}

lldb::ValueObjectSP
ValueObject::GetChildAtIndexPath (const std::vector<size_t> &idxs,
                                  size_t* index_of_error)
{
    if (idxs.size() == 0)
        return GetSP();
    ValueObjectSP root(GetSP());
    for (size_t idx : idxs)
    {
        root = root->GetChildAtIndex(idx, true);
        if (!root)
        {
            if (index_of_error)
                *index_of_error = idx;
            return root;
        }
    }
    return root;
}

lldb::ValueObjectSP
ValueObject::GetChildAtIndexPath (const std::vector< std::pair<size_t, bool> > &idxs,
                                  size_t* index_of_error)
{
    if (idxs.size() == 0)
        return GetSP();
    ValueObjectSP root(GetSP());
    for (std::pair<size_t, bool> idx : idxs)
    {
        root = root->GetChildAtIndex(idx.first, idx.second);
        if (!root)
        {
            if (index_of_error)
                *index_of_error = idx.first;
            return root;
        }
    }
    return root;
}

size_t
ValueObject::GetIndexOfChildWithName (const ConstString &name)
{
    bool omit_empty_base_classes = true;
    return ClangASTContext::GetIndexOfChildWithName (GetClangAST(),
                                                     GetClangType(),
                                                     name.GetCString(),
                                                     omit_empty_base_classes);
}

ValueObjectSP
ValueObject::GetChildMemberWithName (const ConstString &name, bool can_create)
{
    // when getting a child by name, it could be buried inside some base
    // classes (which really aren't part of the expression path), so we
    // need a vector of indexes that can get us down to the correct child
    ValueObjectSP child_sp;

    // We may need to update our value if we are dynamic
    if (IsPossibleDynamicType ())
        UpdateValueIfNeeded(false);

    std::vector<uint32_t> child_indexes;
    clang::ASTContext *clang_ast = GetClangAST();
    void *clang_type = GetClangType();
    bool omit_empty_base_classes = true;
    const size_t num_child_indexes =  ClangASTContext::GetIndexOfChildMemberWithName (clang_ast,
                                                                                      clang_type,
                                                                                      name.GetCString(),
                                                                                      omit_empty_base_classes,
                                                                                      child_indexes);
    if (num_child_indexes > 0)
    {
        std::vector<uint32_t>::const_iterator pos = child_indexes.begin ();
        std::vector<uint32_t>::const_iterator end = child_indexes.end ();

        child_sp = GetChildAtIndex(*pos, can_create);
        for (++pos; pos != end; ++pos)
        {
            if (child_sp)
            {
                ValueObjectSP new_child_sp(child_sp->GetChildAtIndex (*pos, can_create));
                child_sp = new_child_sp;
            }
            else
            {
                child_sp.reset();
            }

        }
    }
    return child_sp;
}


size_t
ValueObject::GetNumChildren ()
{
    UpdateValueIfNeeded();
    if (!m_children_count_valid)
    {
        SetNumChildren (CalculateNumChildren());
    }
    return m_children.GetChildrenCount();
}

bool
ValueObject::MightHaveChildren()
{
    bool has_children = false;
    const uint32_t type_info = GetTypeInfo();
    if (type_info)
    {
        if (type_info & (ClangASTContext::eTypeHasChildren |
                         ClangASTContext::eTypeIsPointer |
                         ClangASTContext::eTypeIsReference))
            has_children = true;
    }
    else
    {
        has_children = GetNumChildren () > 0;
    }
    return has_children;
}

// Should only be called by ValueObject::GetNumChildren()
void
ValueObject::SetNumChildren (size_t num_children)
{
    m_children_count_valid = true;
    m_children.SetChildrenCount(num_children);
}

void
ValueObject::SetName (const ConstString &name)
{
    m_name = name;
}

ValueObject *
ValueObject::CreateChildAtIndex (size_t idx, bool synthetic_array_member, int32_t synthetic_index)
{
    ValueObject *valobj = NULL;
    
    bool omit_empty_base_classes = true;
    bool ignore_array_bounds = synthetic_array_member;
    std::string child_name_str;
    uint32_t child_byte_size = 0;
    int32_t child_byte_offset = 0;
    uint32_t child_bitfield_bit_size = 0;
    uint32_t child_bitfield_bit_offset = 0;
    bool child_is_base_class = false;
    bool child_is_deref_of_parent = false;

    const bool transparent_pointers = synthetic_array_member == false;
    clang::ASTContext *clang_ast = GetClangAST();
    clang_type_t clang_type = GetClangType();
    clang_type_t child_clang_type;
    
    ExecutionContext exe_ctx (GetExecutionContextRef());
    
    child_clang_type = ClangASTContext::GetChildClangTypeAtIndex (&exe_ctx,
                                                                  clang_ast,
                                                                  GetName().GetCString(),
                                                                  clang_type,
                                                                  idx,
                                                                  transparent_pointers,
                                                                  omit_empty_base_classes,
                                                                  ignore_array_bounds,
                                                                  child_name_str,
                                                                  child_byte_size,
                                                                  child_byte_offset,
                                                                  child_bitfield_bit_size,
                                                                  child_bitfield_bit_offset,
                                                                  child_is_base_class,
                                                                  child_is_deref_of_parent);
    if (child_clang_type)
    {
        if (synthetic_index)
            child_byte_offset += child_byte_size * synthetic_index;

        ConstString child_name;
        if (!child_name_str.empty())
            child_name.SetCString (child_name_str.c_str());

        valobj = new ValueObjectChild (*this,
                                       clang_ast,
                                       child_clang_type,
                                       child_name,
                                       child_byte_size,
                                       child_byte_offset,
                                       child_bitfield_bit_size,
                                       child_bitfield_bit_offset,
                                       child_is_base_class,
                                       child_is_deref_of_parent,
                                       eAddressTypeInvalid);
        //if (valobj)
        //    valobj->SetAddressTypeOfChildren(eAddressTypeInvalid);
   }
    
    return valobj;
}

bool
ValueObject::GetSummaryAsCString (TypeSummaryImpl* summary_ptr,
                                  std::string& destination)
{
    destination.clear();

    // ideally we would like to bail out if passing NULL, but if we do so
    // we end up not providing the summary for function pointers anymore
    if (/*summary_ptr == NULL ||*/ m_is_getting_summary)
        return false;
    
    m_is_getting_summary = true;
    
    // this is a hot path in code and we prefer to avoid setting this string all too often also clearing out other
    // information that we might care to see in a crash log. might be useful in very specific situations though.
    /*Host::SetCrashDescriptionWithFormat("Trying to fetch a summary for %s %s. Summary provider's description is %s",
                                        GetTypeName().GetCString(),
                                        GetName().GetCString(),
                                        summary_ptr->GetDescription().c_str());*/
    
    if (UpdateValueIfNeeded (false))
    {
        if (summary_ptr)
        {
            if (HasSyntheticValue())
                m_synthetic_value->UpdateValueIfNeeded(); // the summary might depend on the synthetic children being up-to-date (e.g. ${svar%#})
            summary_ptr->FormatObject(this, destination);
        }
        else
        {
            clang_type_t clang_type = GetClangType();
            
            // Do some default printout for function pointers
            if (clang_type)
            {
                StreamString sstr;
                clang_type_t elem_or_pointee_clang_type;
                const Flags type_flags (ClangASTContext::GetTypeInfo (clang_type, 
                                                                      GetClangAST(), 
                                                                      &elem_or_pointee_clang_type));
                
                if (ClangASTContext::IsFunctionPointerType (clang_type))
                {
                    AddressType func_ptr_address_type = eAddressTypeInvalid;
                    addr_t func_ptr_address = GetPointerValue (&func_ptr_address_type);
                    if (func_ptr_address != 0 && func_ptr_address != LLDB_INVALID_ADDRESS)
                    {
                        switch (func_ptr_address_type)
                        {
                            case eAddressTypeInvalid:
                            case eAddressTypeFile:
                                break;
                                
                            case eAddressTypeLoad:
                            {
                                ExecutionContext exe_ctx (GetExecutionContextRef());
                                
                                Address so_addr;
                                Target *target = exe_ctx.GetTargetPtr();
                                if (target && target->GetSectionLoadList().IsEmpty() == false)
                                {
                                    if (target->GetSectionLoadList().ResolveLoadAddress(func_ptr_address, so_addr))
                                    {
                                        so_addr.Dump (&sstr, 
                                                      exe_ctx.GetBestExecutionContextScope(), 
                                                      Address::DumpStyleResolvedDescription, 
                                                      Address::DumpStyleSectionNameOffset);
                                    }
                                }
                            }
                                break;
                                
                            case eAddressTypeHost:
                                break;
                        }
                    }
                    if (sstr.GetSize() > 0)
                    {
                        destination.assign (1, '(');
                        destination.append (sstr.GetData(), sstr.GetSize());
                        destination.append (1, ')');
                    }
                }
            }
        }
    }
    m_is_getting_summary = false;
    return !destination.empty();
}

const char *
ValueObject::GetSummaryAsCString ()
{
    if (UpdateValueIfNeeded(true) && m_summary_str.empty())
    {
        GetSummaryAsCString(GetSummaryFormat().get(),
                            m_summary_str);
    }
    if (m_summary_str.empty())
        return NULL;
    return m_summary_str.c_str();
}

bool
ValueObject::IsCStringContainer(bool check_pointer)
{
    clang_type_t elem_or_pointee_clang_type;
    const Flags type_flags (GetTypeInfo (&elem_or_pointee_clang_type));
    bool is_char_arr_ptr (type_flags.AnySet (ClangASTContext::eTypeIsArray | ClangASTContext::eTypeIsPointer) &&
                          ClangASTContext::IsCharType (elem_or_pointee_clang_type));
    if (!is_char_arr_ptr)
        return false;
    if (!check_pointer)
        return true;
    if (type_flags.Test(ClangASTContext::eTypeIsArray))
        return true;
    addr_t cstr_address = LLDB_INVALID_ADDRESS;
    AddressType cstr_address_type = eAddressTypeInvalid;
    cstr_address = GetAddressOf (true, &cstr_address_type);
    return (cstr_address != LLDB_INVALID_ADDRESS);
}

size_t
ValueObject::GetPointeeData (DataExtractor& data,
                             uint32_t item_idx,
                             uint32_t item_count)
{
    clang_type_t pointee_or_element_clang_type;
    const uint32_t type_info = GetTypeInfo (&pointee_or_element_clang_type);
    const bool is_pointer_type = type_info & ClangASTContext::eTypeIsPointer;
    const bool is_array_type = type_info & ClangASTContext::eTypeIsArray;
    if (!(is_pointer_type || is_array_type))
        return 0;
    
    if (item_count == 0)
        return 0;
    
    clang::ASTContext *ast = GetClangAST();
    ClangASTType pointee_or_element_type(ast, pointee_or_element_clang_type);
    
    const uint64_t item_type_size = pointee_or_element_type.GetClangTypeByteSize();
    
    const uint64_t bytes = item_count * item_type_size;
    
    const uint64_t offset = item_idx * item_type_size;
    
    if (item_idx == 0 && item_count == 1) // simply a deref
    {
        if (is_pointer_type)
        {
            Error error;
            ValueObjectSP pointee_sp = Dereference(error);
            if (error.Fail() || pointee_sp.get() == NULL)
                return 0;
            return pointee_sp->GetDataExtractor().Copy(data);
        }
        else
        {
            ValueObjectSP child_sp = GetChildAtIndex(0, true);
            if (child_sp.get() == NULL)
                return 0;
            return child_sp->GetDataExtractor().Copy(data);
        }
        return true;
    }
    else /* (items > 1) */
    {
        Error error;
        lldb_private::DataBufferHeap* heap_buf_ptr = NULL;
        lldb::DataBufferSP data_sp(heap_buf_ptr = new lldb_private::DataBufferHeap());
        
        AddressType addr_type;
        lldb::addr_t addr = is_pointer_type ? GetPointerValue(&addr_type) : GetAddressOf(true, &addr_type);
        
        switch (addr_type)
        {
            case eAddressTypeFile:
                {
                    ModuleSP module_sp (GetModule());
                    if (module_sp)
                    {
                        addr = addr + offset;
                        Address so_addr;
                        module_sp->ResolveFileAddress(addr, so_addr);
                        ExecutionContext exe_ctx (GetExecutionContextRef());
                        Target* target = exe_ctx.GetTargetPtr();
                        if (target)
                        {
                            heap_buf_ptr->SetByteSize(bytes);
                            size_t bytes_read = target->ReadMemory(so_addr, false, heap_buf_ptr->GetBytes(), bytes, error);
                            if (error.Success())
                            {
                                data.SetData(data_sp);
                                return bytes_read;
                            }
                        }
                    }
                }
                break;
            case eAddressTypeLoad:
                {
                    ExecutionContext exe_ctx (GetExecutionContextRef());
                    Process *process = exe_ctx.GetProcessPtr();
                    if (process)
                    {
                        heap_buf_ptr->SetByteSize(bytes);
                        size_t bytes_read = process->ReadMemory(addr + offset, heap_buf_ptr->GetBytes(), bytes, error);
                        if (error.Success())
                        {
                            data.SetData(data_sp);
                            return bytes_read;
                        }
                    }
                }
                break;
            case eAddressTypeHost:
                {
                    ClangASTType valobj_type(ast, GetClangType());
                    uint64_t max_bytes = valobj_type.GetClangTypeByteSize();
                    if (max_bytes > offset)
                    {
                        size_t bytes_read = std::min<uint64_t>(max_bytes - offset, bytes);
                        heap_buf_ptr->CopyData((uint8_t*)(addr + offset), bytes_read);
                        data.SetData(data_sp);
                        return bytes_read;
                    }
                }
                break;
            case eAddressTypeInvalid:
                break;
        }
    }
    return 0;
}

uint64_t
ValueObject::GetData (DataExtractor& data)
{
    UpdateValueIfNeeded(false);
    ExecutionContext exe_ctx (GetExecutionContextRef());
    Error error = m_value.GetValueAsData(&exe_ctx, GetClangAST(), data, 0, GetModule().get());
    if (error.Fail())
    {
        if (m_data.GetByteSize())
        {
            data = m_data;
            return data.GetByteSize();
        }
        else
        {
            return 0;
        }
    }
    data.SetAddressByteSize(m_data.GetAddressByteSize());
    data.SetByteOrder(m_data.GetByteOrder());
    return data.GetByteSize();
}

bool
ValueObject::SetData (DataExtractor &data, Error &error)
{
    error.Clear();
    // Make sure our value is up to date first so that our location and location
    // type is valid.
    if (!UpdateValueIfNeeded(false))
    {
        error.SetErrorString("unable to read value");
        return false;
    }
    
    uint64_t count = 0;
    Encoding encoding = ClangASTType::GetEncoding (GetClangType(), count);
    
    const size_t byte_size = GetByteSize();
    
    Value::ValueType value_type = m_value.GetValueType();
    
    switch (value_type)
    {
    case Value::eValueTypeScalar:
        {
            Error set_error = m_value.GetScalar().SetValueFromData(data, encoding, byte_size);
            
            if (!set_error.Success())
            {
                error.SetErrorStringWithFormat("unable to set scalar value: %s", set_error.AsCString());
                return false;
            }
        }
        break;
    case Value::eValueTypeLoadAddress:
        {
            // If it is a load address, then the scalar value is the storage location
            // of the data, and we have to shove this value down to that load location.
            ExecutionContext exe_ctx (GetExecutionContextRef());
            Process *process = exe_ctx.GetProcessPtr();
            if (process)
            {
                addr_t target_addr = m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
                size_t bytes_written = process->WriteMemory(target_addr,
                                                            data.GetDataStart(),
                                                            byte_size,
                                                            error);
                if (!error.Success())
                    return false;
                if (bytes_written != byte_size)
                {
                    error.SetErrorString("unable to write value to memory");
                    return false;
                }
            }
        }
        break;
    case Value::eValueTypeHostAddress:
        {
            // If it is a host address, then we stuff the scalar as a DataBuffer into the Value's data.            
            DataBufferSP buffer_sp (new DataBufferHeap(byte_size, 0));
            m_data.SetData(buffer_sp, 0);
            data.CopyByteOrderedData (0,
                                      byte_size,
                                      const_cast<uint8_t *>(m_data.GetDataStart()),
                                      byte_size,
                                      m_data.GetByteOrder());
            m_value.GetScalar() = (uintptr_t)m_data.GetDataStart();
        }
        break;
    case Value::eValueTypeFileAddress:
    case Value::eValueTypeVector:
        break;
    }
    
    // If we have reached this point, then we have successfully changed the value.
    SetNeedsUpdate();
    return true;
}

// will compute strlen(str), but without consuming more than
// maxlen bytes out of str (this serves the purpose of reading
// chunks of a string without having to worry about
// missing NULL terminators in the chunk)
// of course, if strlen(str) > maxlen, the function will return
// maxlen_value (which should be != maxlen, because that allows you
// to know whether strlen(str) == maxlen or strlen(str) > maxlen)
static uint32_t
strlen_or_inf (const char* str,
               uint32_t maxlen,
               uint32_t maxlen_value)
{
    uint32_t len = 0;
    if (str)
    {
        while(*str)
        {
            len++;str++;
            if (len >= maxlen)
                return maxlen_value;
        }
    }
    return len;
}

size_t
ValueObject::ReadPointedString (Stream& s,
                                Error& error,
                                uint32_t max_length,
                                bool honor_array,
                                Format item_format)
{
    ExecutionContext exe_ctx (GetExecutionContextRef());
    Target* target = exe_ctx.GetTargetPtr();

    if (!target)
    {
        s << "<no target to read from>";
        error.SetErrorString("no target to read from");
        return 0;
    }
    
    if (max_length == 0)
        max_length = target->GetMaximumSizeOfStringSummary();
    
    size_t bytes_read = 0;
    size_t total_bytes_read = 0;
    
    clang_type_t clang_type = GetClangType();
    clang_type_t elem_or_pointee_clang_type;
    const Flags type_flags (GetTypeInfo (&elem_or_pointee_clang_type));
    if (type_flags.AnySet (ClangASTContext::eTypeIsArray | ClangASTContext::eTypeIsPointer) &&
        ClangASTContext::IsCharType (elem_or_pointee_clang_type))
    {
        addr_t cstr_address = LLDB_INVALID_ADDRESS;
        AddressType cstr_address_type = eAddressTypeInvalid;
        
        size_t cstr_len = 0;
        bool capped_data = false;
        if (type_flags.Test (ClangASTContext::eTypeIsArray))
        {
            // We have an array
            cstr_len = ClangASTContext::GetArraySize (clang_type);
            if (cstr_len > max_length)
            {
                capped_data = true;
                cstr_len = max_length;
            }
            cstr_address = GetAddressOf (true, &cstr_address_type);
        }
        else
        {
            // We have a pointer
            cstr_address = GetPointerValue (&cstr_address_type);
        }
        
        if (cstr_address == 0 || cstr_address == LLDB_INVALID_ADDRESS)
        {
            s << "<invalid address>";
            error.SetErrorString("invalid address");
            return 0;
        }

        Address cstr_so_addr (cstr_address);
        DataExtractor data;
        if (cstr_len > 0 && honor_array)
        {
            // I am using GetPointeeData() here to abstract the fact that some ValueObjects are actually frozen pointers in the host
            // but the pointed-to data lives in the debuggee, and GetPointeeData() automatically takes care of this
            GetPointeeData(data, 0, cstr_len);

            if ((bytes_read = data.GetByteSize()) > 0)
            {
                total_bytes_read = bytes_read;
                s << '"';
                data.Dump (&s,
                           0,                 // Start offset in "data"
                           item_format,
                           1,                 // Size of item (1 byte for a char!)
                           bytes_read,        // How many bytes to print?
                           UINT32_MAX,        // num per line
                           LLDB_INVALID_ADDRESS,// base address
                           0,                 // bitfield bit size
                           0);                // bitfield bit offset
                if (capped_data)
                    s << "...";
                s << '"';
            }
        }
        else
        {
            cstr_len = max_length;
            const size_t k_max_buf_size = 64;
                                        
            size_t offset = 0;
            
            int cstr_len_displayed = -1;
            bool capped_cstr = false;
            // I am using GetPointeeData() here to abstract the fact that some ValueObjects are actually frozen pointers in the host
            // but the pointed-to data lives in the debuggee, and GetPointeeData() automatically takes care of this
            while ((bytes_read = GetPointeeData(data, offset, k_max_buf_size)) > 0)
            {
                total_bytes_read += bytes_read;
                const char *cstr = data.PeekCStr(0);
                size_t len = strlen_or_inf (cstr, k_max_buf_size, k_max_buf_size+1);
                if (len > k_max_buf_size)
                    len = k_max_buf_size;
                if (cstr && cstr_len_displayed < 0)
                    s << '"';

                if (cstr_len_displayed < 0)
                    cstr_len_displayed = len;

                if (len == 0)
                    break;
                cstr_len_displayed += len;
                if (len > bytes_read)
                    len = bytes_read;
                if (len > cstr_len)
                    len = cstr_len;
                
                data.Dump (&s,
                           0,                 // Start offset in "data"
                           item_format,
                           1,                 // Size of item (1 byte for a char!)
                           len,               // How many bytes to print?
                           UINT32_MAX,        // num per line
                           LLDB_INVALID_ADDRESS,// base address
                           0,                 // bitfield bit size
                           0);                // bitfield bit offset
                
                if (len < k_max_buf_size)
                    break;
                
                if (len >= cstr_len)
                {
                    capped_cstr = true;
                    break;
                }

                cstr_len -= len;
                offset += len;
            }
            
            if (cstr_len_displayed >= 0)
            {
                s << '"';
                if (capped_cstr)
                    s << "...";
            }
        }
    }
    else
    {
        error.SetErrorString("not a string object");
        s << "<not a string object>";
    }
    return total_bytes_read;
}

const char *
ValueObject::GetObjectDescription ()
{
    
    if (!UpdateValueIfNeeded (true))
        return NULL;

    if (!m_object_desc_str.empty())
        return m_object_desc_str.c_str();

    ExecutionContext exe_ctx (GetExecutionContextRef());
    Process *process = exe_ctx.GetProcessPtr();
    if (process == NULL)
        return NULL;
        
    StreamString s;
    
    LanguageType language = GetObjectRuntimeLanguage();
    LanguageRuntime *runtime = process->GetLanguageRuntime(language);
    
    if (runtime == NULL)
    {
        // Aw, hell, if the things a pointer, or even just an integer, let's try ObjC anyway...
        clang_type_t opaque_qual_type = GetClangType();
        if (opaque_qual_type != NULL)
        {
            bool is_signed;
            if (ClangASTContext::IsIntegerType (opaque_qual_type, is_signed) 
                || ClangASTContext::IsPointerType (opaque_qual_type))
            {
                runtime = process->GetLanguageRuntime(eLanguageTypeObjC);
            }
        }
    }
    
    if (runtime && runtime->GetObjectDescription(s, *this))
    {
        m_object_desc_str.append (s.GetData());
    }
    
    if (m_object_desc_str.empty())
        return NULL;
    else
        return m_object_desc_str.c_str();
}

bool
ValueObject::GetValueAsCString (lldb::Format format,
                                std::string& destination)
{
    if (ClangASTContext::IsAggregateType (GetClangType()) == false &&
        UpdateValueIfNeeded(false))
    {
        const Value::ContextType context_type = m_value.GetContextType();
        
        switch (context_type)
        {
            case Value::eContextTypeClangType:
            case Value::eContextTypeLLDBType:
            case Value::eContextTypeVariable:
            {
                clang_type_t clang_type = GetClangType ();
                if (clang_type)
                {
                     // put custom bytes to display in this DataExtractor to override the default value logic
                    lldb_private::DataExtractor special_format_data;
                    clang::ASTContext* ast = GetClangAST();
                    if (format == eFormatCString)
                    {
                        Flags type_flags(ClangASTContext::GetTypeInfo(clang_type, ast, NULL));
                        if (type_flags.Test(ClangASTContext::eTypeIsPointer) && !type_flags.Test(ClangASTContext::eTypeIsObjC))
                        {
                            // if we are dumping a pointer as a c-string, get the pointee data as a string
                            TargetSP target_sp(GetTargetSP());
                            if (target_sp)
                            {
                                size_t max_len = target_sp->GetMaximumSizeOfStringSummary();
                                Error error;
                                DataBufferSP buffer_sp(new DataBufferHeap(max_len+1,0));
                                Address address(GetPointerValue());
                                if (target_sp->ReadCStringFromMemory(address, (char*)buffer_sp->GetBytes(), max_len, error) && error.Success())
                                    special_format_data.SetData(buffer_sp);
                            }
                        }
                    }
                    
                    StreamString sstr;
                    ExecutionContext exe_ctx (GetExecutionContextRef());
                    ClangASTType::DumpTypeValue (ast,                           // The clang AST
                                                 clang_type,                    // The clang type to display
                                                 &sstr,                         // The stream to use for display
                                                 format,                        // Format to display this type with
                                                 special_format_data.GetByteSize() ?
                                                 special_format_data: m_data,   // Data to extract from
                                                 0,                             // Byte offset into "m_data"
                                                 GetByteSize(),                 // Byte size of item in "m_data"
                                                 GetBitfieldBitSize(),          // Bitfield bit size
                                                 GetBitfieldBitOffset(),        // Bitfield bit offset
                                                 exe_ctx.GetBestExecutionContextScope()); 
                    // Don't set the m_error to anything here otherwise
                    // we won't be able to re-format as anything else. The
                    // code for ClangASTType::DumpTypeValue() should always
                    // return something, even if that something contains
                    // an error messsage. "m_error" is used to detect errors
                    // when reading the valid object, not for formatting errors.
                    if (sstr.GetString().empty())
                        destination.clear();
                    else
                        destination.swap(sstr.GetString());
                }
            }
                break;
                
            case Value::eContextTypeRegisterInfo:
            {
                const RegisterInfo *reg_info = m_value.GetRegisterInfo();
                if (reg_info)
                {
                    ExecutionContext exe_ctx (GetExecutionContextRef());
                    
                    StreamString reg_sstr;
                    m_data.Dump (&reg_sstr, 
                                 0, 
                                 format, 
                                 reg_info->byte_size, 
                                 1, 
                                 UINT32_MAX, 
                                 LLDB_INVALID_ADDRESS, 
                                 0, 
                                 0, 
                                 exe_ctx.GetBestExecutionContextScope());
                    destination.swap(reg_sstr.GetString());
                }
            }
                break;
                
            default:
                break;
        }
        return !destination.empty();
    }
    else
        return false;
}

const char *
ValueObject::GetValueAsCString ()
{
    if (UpdateValueIfNeeded(true) && m_value_str.empty())
    {
        lldb::Format my_format = GetFormat();
        if (my_format == lldb::eFormatDefault)
        {
            if (m_type_format_sp)
                my_format = m_type_format_sp->GetFormat();
            else
            {
                if (m_is_bitfield_for_scalar)
                    my_format = eFormatUnsigned;
                else
                {
                    if (m_value.GetContextType() == Value::eContextTypeRegisterInfo)
                    {
                        const RegisterInfo *reg_info = m_value.GetRegisterInfo();
                        if (reg_info)
                            my_format = reg_info->format;
                    }
                    else
                    {
                        clang_type_t clang_type = GetClangType ();
                        my_format = ClangASTType::GetFormat(clang_type);
                    }
                }
            }
        }
        if (GetValueAsCString(my_format, m_value_str))
        {
            if (!m_value_did_change && m_old_value_valid)
            {
                // The value was gotten successfully, so we consider the
                // value as changed if the value string differs
                SetValueDidChange (m_old_value_str != m_value_str);
            }
        }
    }
    if (m_value_str.empty())
        return NULL;
    return m_value_str.c_str();
}

// if > 8bytes, 0 is returned. this method should mostly be used
// to read address values out of pointers
uint64_t
ValueObject::GetValueAsUnsigned (uint64_t fail_value, bool *success)
{
    // If our byte size is zero this is an aggregate type that has children
    if (ClangASTContext::IsAggregateType (GetClangType()) == false)
    {
        Scalar scalar;
        if (ResolveValue (scalar))
        {
            if (success)
                *success = true;
            return scalar.ULongLong(fail_value);
        }
        // fallthrough, otherwise...
    }

    if (success)
        *success = false;
    return fail_value;
}

// if any more "special cases" are added to ValueObject::DumpPrintableRepresentation() please keep
// this call up to date by returning true for your new special cases. We will eventually move
// to checking this call result before trying to display special cases
bool
ValueObject::HasSpecialPrintableRepresentation(ValueObjectRepresentationStyle val_obj_display,
                                               Format custom_format)
{
    clang_type_t elem_or_pointee_type;
    Flags flags(GetTypeInfo(&elem_or_pointee_type));
    
    if (flags.AnySet(ClangASTContext::eTypeIsArray | ClangASTContext::eTypeIsPointer)
        && val_obj_display == ValueObject::eValueObjectRepresentationStyleValue)
    {        
        if (IsCStringContainer(true) && 
            (custom_format == eFormatCString ||
             custom_format == eFormatCharArray ||
             custom_format == eFormatChar ||
             custom_format == eFormatVectorOfChar))
            return true;

        if (flags.Test(ClangASTContext::eTypeIsArray))
        {
            if ((custom_format == eFormatBytes) ||
                (custom_format == eFormatBytesWithASCII))
                return true;
            
            if ((custom_format == eFormatVectorOfChar) ||
                (custom_format == eFormatVectorOfFloat32) ||
                (custom_format == eFormatVectorOfFloat64) ||
                (custom_format == eFormatVectorOfSInt16) ||
                (custom_format == eFormatVectorOfSInt32) ||
                (custom_format == eFormatVectorOfSInt64) ||
                (custom_format == eFormatVectorOfSInt8) ||
                (custom_format == eFormatVectorOfUInt128) ||
                (custom_format == eFormatVectorOfUInt16) ||
                (custom_format == eFormatVectorOfUInt32) ||
                (custom_format == eFormatVectorOfUInt64) ||
                (custom_format == eFormatVectorOfUInt8))
                return true;
        }
    }
    return false;
}

bool
ValueObject::DumpPrintableRepresentation(Stream& s,
                                         ValueObjectRepresentationStyle val_obj_display,
                                         Format custom_format,
                                         PrintableRepresentationSpecialCases special)
{

    clang_type_t elem_or_pointee_type;
    Flags flags(GetTypeInfo(&elem_or_pointee_type));
    
    bool allow_special = ((special & ePrintableRepresentationSpecialCasesAllow) == ePrintableRepresentationSpecialCasesAllow);
    bool only_special = ((special & ePrintableRepresentationSpecialCasesOnly) == ePrintableRepresentationSpecialCasesOnly);
    
    if (allow_special)
    {
        if (flags.AnySet(ClangASTContext::eTypeIsArray | ClangASTContext::eTypeIsPointer)
             && val_obj_display == ValueObject::eValueObjectRepresentationStyleValue)
        {
            // when being asked to get a printable display an array or pointer type directly, 
            // try to "do the right thing"
            
            if (IsCStringContainer(true) && 
                (custom_format == eFormatCString ||
                 custom_format == eFormatCharArray ||
                 custom_format == eFormatChar ||
                 custom_format == eFormatVectorOfChar)) // print char[] & char* directly
            {
                Error error;
                ReadPointedString(s,
                                  error,
                                  0,
                                  (custom_format == eFormatVectorOfChar) ||
                                  (custom_format == eFormatCharArray));
                return !error.Fail();
            }
            
            if (custom_format == eFormatEnum)
                return false;
            
            // this only works for arrays, because I have no way to know when
            // the pointed memory ends, and no special \0 end of data marker
            if (flags.Test(ClangASTContext::eTypeIsArray))
            {
                if ((custom_format == eFormatBytes) ||
                    (custom_format == eFormatBytesWithASCII))
                {
                    const size_t count = GetNumChildren();
                                    
                    s << '[';
                    for (size_t low = 0; low < count; low++)
                    {
                        
                        if (low)
                            s << ',';
                        
                        ValueObjectSP child = GetChildAtIndex(low,true);
                        if (!child.get())
                        {
                            s << "<invalid child>";
                            continue;
                        }
                        child->DumpPrintableRepresentation(s, ValueObject::eValueObjectRepresentationStyleValue, custom_format);
                    }                
                    
                    s << ']';
                    
                    return true;
                }
                
                if ((custom_format == eFormatVectorOfChar) ||
                    (custom_format == eFormatVectorOfFloat32) ||
                    (custom_format == eFormatVectorOfFloat64) ||
                    (custom_format == eFormatVectorOfSInt16) ||
                    (custom_format == eFormatVectorOfSInt32) ||
                    (custom_format == eFormatVectorOfSInt64) ||
                    (custom_format == eFormatVectorOfSInt8) ||
                    (custom_format == eFormatVectorOfUInt128) ||
                    (custom_format == eFormatVectorOfUInt16) ||
                    (custom_format == eFormatVectorOfUInt32) ||
                    (custom_format == eFormatVectorOfUInt64) ||
                    (custom_format == eFormatVectorOfUInt8)) // arrays of bytes, bytes with ASCII or any vector format should be printed directly
                {
                    const size_t count = GetNumChildren();

                    Format format = FormatManager::GetSingleItemFormat(custom_format);
                    
                    s << '[';
                    for (size_t low = 0; low < count; low++)
                    {
                        
                        if (low)
                            s << ',';
                        
                        ValueObjectSP child = GetChildAtIndex(low,true);
                        if (!child.get())
                        {
                            s << "<invalid child>";
                            continue;
                        }
                        child->DumpPrintableRepresentation(s, ValueObject::eValueObjectRepresentationStyleValue, format);
                    }                
                    
                    s << ']';
                    
                    return true;
                }
            }
            
            if ((custom_format == eFormatBoolean) ||
                (custom_format == eFormatBinary) ||
                (custom_format == eFormatChar) ||
                (custom_format == eFormatCharPrintable) ||
                (custom_format == eFormatComplexFloat) ||
                (custom_format == eFormatDecimal) ||
                (custom_format == eFormatHex) ||
                (custom_format == eFormatHexUppercase) ||
                (custom_format == eFormatFloat) ||
                (custom_format == eFormatOctal) ||
                (custom_format == eFormatOSType) ||
                (custom_format == eFormatUnicode16) ||
                (custom_format == eFormatUnicode32) ||
                (custom_format == eFormatUnsigned) ||
                (custom_format == eFormatPointer) ||
                (custom_format == eFormatComplexInteger) ||
                (custom_format == eFormatComplex) ||
                (custom_format == eFormatDefault)) // use the [] operator
                return false;
        }
    }
    
    if (only_special)
        return false;
    
    bool var_success = false;
    
    {
        const char *cstr = NULL;
        StreamString strm;

        if (custom_format != eFormatInvalid)
            SetFormat(custom_format);
        
        switch(val_obj_display)
        {
            case eValueObjectRepresentationStyleValue:
                cstr = GetValueAsCString();
                break;
                
            case eValueObjectRepresentationStyleSummary:
                cstr = GetSummaryAsCString();
                break;
                
            case eValueObjectRepresentationStyleLanguageSpecific:
                cstr = GetObjectDescription();
                break;
                
            case eValueObjectRepresentationStyleLocation:
                cstr = GetLocationAsCString();
                break;
                
            case eValueObjectRepresentationStyleChildrenCount:
                strm.Printf("%zu", GetNumChildren());
                cstr = strm.GetString().c_str();
                break;
                
            case eValueObjectRepresentationStyleType:
                cstr = GetTypeName().AsCString();
                break;
        }
        
        if (!cstr)
        {
            if (val_obj_display == eValueObjectRepresentationStyleValue)
                cstr = GetSummaryAsCString();
            else if (val_obj_display == eValueObjectRepresentationStyleSummary)
            {
                if (ClangASTContext::IsAggregateType (GetClangType()) == true)
                {
                    strm.Printf("%s @ %s", GetTypeName().AsCString(), GetLocationAsCString());
                    cstr = strm.GetString().c_str();
                }
                else
                    cstr = GetValueAsCString();
            }
        }
        
        if (cstr)
            s.PutCString(cstr);
        else
        {
            if (m_error.Fail())
                s.Printf("<%s>", m_error.AsCString());
            else if (val_obj_display == eValueObjectRepresentationStyleSummary)
                s.PutCString("<no summary available>");
            else if (val_obj_display == eValueObjectRepresentationStyleValue)
                s.PutCString("<no value available>");
            else if (val_obj_display == eValueObjectRepresentationStyleLanguageSpecific)
                s.PutCString("<not a valid Objective-C object>"); // edit this if we have other runtimes that support a description
            else
                s.PutCString("<no printable representation>");
        }
        
        // we should only return false here if we could not do *anything*
        // even if we have an error message as output, that's a success
        // from our callers' perspective, so return true
        var_success = true;
        
        if (custom_format != eFormatInvalid)
            SetFormat(eFormatDefault);
    }
    
    return var_success;
}

addr_t
ValueObject::GetAddressOf (bool scalar_is_load_address, AddressType *address_type)
{
    if (!UpdateValueIfNeeded(false))
        return LLDB_INVALID_ADDRESS;
        
    switch (m_value.GetValueType())
    {
    case Value::eValueTypeScalar:
    case Value::eValueTypeVector:
        if (scalar_is_load_address)
        {
            if(address_type)
                *address_type = eAddressTypeLoad;
            return m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
        }
        break;

    case Value::eValueTypeLoadAddress: 
    case Value::eValueTypeFileAddress:
    case Value::eValueTypeHostAddress:
        {
            if(address_type)
                *address_type = m_value.GetValueAddressType ();
            return m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
        }
        break;
    }
    if (address_type)
        *address_type = eAddressTypeInvalid;
    return LLDB_INVALID_ADDRESS;
}

addr_t
ValueObject::GetPointerValue (AddressType *address_type)
{
    addr_t address = LLDB_INVALID_ADDRESS;
    if(address_type)
        *address_type = eAddressTypeInvalid;
    
    if (!UpdateValueIfNeeded(false))
        return address;
        
    switch (m_value.GetValueType())
    {
    case Value::eValueTypeScalar:
    case Value::eValueTypeVector:
        address = m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
        break;

    case Value::eValueTypeHostAddress:
    case Value::eValueTypeLoadAddress:
    case Value::eValueTypeFileAddress:
        {
            lldb::offset_t data_offset = 0;
            address = m_data.GetPointer(&data_offset);
        }
        break;
    }

    if (address_type)
        *address_type = GetAddressTypeOfChildren();

    return address;
}

bool
ValueObject::SetValueFromCString (const char *value_str, Error& error)
{
    error.Clear();
    // Make sure our value is up to date first so that our location and location
    // type is valid.
    if (!UpdateValueIfNeeded(false))
    {
        error.SetErrorString("unable to read value");
        return false;
    }

    uint64_t count = 0;
    Encoding encoding = ClangASTType::GetEncoding (GetClangType(), count);

    const size_t byte_size = GetByteSize();

    Value::ValueType value_type = m_value.GetValueType();
    
    if (value_type == Value::eValueTypeScalar)
    {
        // If the value is already a scalar, then let the scalar change itself:
        m_value.GetScalar().SetValueFromCString (value_str, encoding, byte_size);
    }
    else if (byte_size <= Scalar::GetMaxByteSize())
    {
        // If the value fits in a scalar, then make a new scalar and again let the
        // scalar code do the conversion, then figure out where to put the new value.
        Scalar new_scalar;
        error = new_scalar.SetValueFromCString (value_str, encoding, byte_size);
        if (error.Success())
        {
            switch (value_type)
            {
            case Value::eValueTypeLoadAddress:
                {
                    // If it is a load address, then the scalar value is the storage location
                    // of the data, and we have to shove this value down to that load location.
                    ExecutionContext exe_ctx (GetExecutionContextRef());
                    Process *process = exe_ctx.GetProcessPtr();
                    if (process)
                    {
                        addr_t target_addr = m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
                        size_t bytes_written = process->WriteScalarToMemory (target_addr, 
                                                                             new_scalar, 
                                                                             byte_size, 
                                                                             error);
                        if (!error.Success())
                            return false;
                        if (bytes_written != byte_size)
                        {
                            error.SetErrorString("unable to write value to memory");
                            return false;
                        }
                    }
                }
                break;
            case Value::eValueTypeHostAddress:
                {
                    // If it is a host address, then we stuff the scalar as a DataBuffer into the Value's data.
                    DataExtractor new_data;
                    new_data.SetByteOrder (m_data.GetByteOrder());
                    
                    DataBufferSP buffer_sp (new DataBufferHeap(byte_size, 0));
                    m_data.SetData(buffer_sp, 0);
                    bool success = new_scalar.GetData(new_data);
                    if (success)
                    {
                        new_data.CopyByteOrderedData (0, 
                                                      byte_size, 
                                                      const_cast<uint8_t *>(m_data.GetDataStart()), 
                                                      byte_size, 
                                                      m_data.GetByteOrder());
                    }
                    m_value.GetScalar() = (uintptr_t)m_data.GetDataStart();
                    
                }
                break;
            case Value::eValueTypeFileAddress:
            case Value::eValueTypeScalar:
            case Value::eValueTypeVector:
                break;
            }
        }
        else
        {
            return false;
        }
    }
    else
    {
        // We don't support setting things bigger than a scalar at present.
        error.SetErrorString("unable to write aggregate data type");
        return false;
    }
    
    // If we have reached this point, then we have successfully changed the value.
    SetNeedsUpdate();
    return true;
}

bool
ValueObject::GetDeclaration (Declaration &decl)
{
    decl.Clear();
    return false;
}

ConstString
ValueObject::GetTypeName()
{
    return ClangASTType::GetConstTypeName (GetClangAST(), GetClangType());
}

ConstString
ValueObject::GetQualifiedTypeName()
{
    return ClangASTType::GetConstQualifiedTypeName (GetClangAST(), GetClangType());
}


LanguageType
ValueObject::GetObjectRuntimeLanguage ()
{
    return ClangASTType::GetMinimumLanguage (GetClangAST(),
                                             GetClangType());
}

void
ValueObject::AddSyntheticChild (const ConstString &key, ValueObject *valobj)
{
    m_synthetic_children[key] = valobj;
}

ValueObjectSP
ValueObject::GetSyntheticChild (const ConstString &key) const
{
    ValueObjectSP synthetic_child_sp;
    std::map<ConstString, ValueObject *>::const_iterator pos = m_synthetic_children.find (key);
    if (pos != m_synthetic_children.end())
        synthetic_child_sp = pos->second->GetSP();
    return synthetic_child_sp;
}

uint32_t
ValueObject::GetTypeInfo (clang_type_t *pointee_or_element_clang_type)
{
    return ClangASTContext::GetTypeInfo (GetClangType(), GetClangAST(), pointee_or_element_clang_type);
}

bool
ValueObject::IsPointerType ()
{
    return ClangASTContext::IsPointerType (GetClangType());
}

bool
ValueObject::IsArrayType ()
{
    return ClangASTContext::IsArrayType (GetClangType(), NULL, NULL, NULL);
}

bool
ValueObject::IsScalarType ()
{
    return ClangASTContext::IsScalarType (GetClangType());
}

bool
ValueObject::IsIntegerType (bool &is_signed)
{
    return ClangASTContext::IsIntegerType (GetClangType(), is_signed);
}

bool
ValueObject::IsPointerOrReferenceType ()
{
    return ClangASTContext::IsPointerOrReferenceType (GetClangType());
}

bool
ValueObject::IsPossibleDynamicType ()
{
    ExecutionContext exe_ctx (GetExecutionContextRef());
    Process *process = exe_ctx.GetProcessPtr();
    if (process)
        return process->IsPossibleDynamicValue(*this);
    else
        return ClangASTContext::IsPossibleDynamicType (GetClangAST (), GetClangType(), NULL, true, true);
}

bool
ValueObject::IsObjCNil ()
{
    const uint32_t mask = ClangASTContext::eTypeIsObjC | ClangASTContext::eTypeIsPointer;
    bool isObjCpointer = ( ((ClangASTContext::GetTypeInfo(GetClangType(), GetClangAST(), NULL)) & mask) == mask);
    if (!isObjCpointer)
        return false;
    bool canReadValue = true;
    bool isZero = GetValueAsUnsigned(0,&canReadValue) == 0;
    return canReadValue && isZero;
}

ValueObjectSP
ValueObject::GetSyntheticArrayMember (size_t index, bool can_create)
{
    const uint32_t type_info = GetTypeInfo ();
    if (type_info & ClangASTContext::eTypeIsArray)
        return GetSyntheticArrayMemberFromArray(index, can_create);

    if (type_info & ClangASTContext::eTypeIsPointer)
        return GetSyntheticArrayMemberFromPointer(index, can_create);
    
    return ValueObjectSP();
    
}

ValueObjectSP
ValueObject::GetSyntheticArrayMemberFromPointer (size_t index, bool can_create)
{
    ValueObjectSP synthetic_child_sp;
    if (IsPointerType ())
    {
        char index_str[64];
        snprintf(index_str, sizeof(index_str), "[%zu]", index);
        ConstString index_const_str(index_str);
        // Check if we have already created a synthetic array member in this
        // valid object. If we have we will re-use it.
        synthetic_child_sp = GetSyntheticChild (index_const_str);
        if (!synthetic_child_sp)
        {
            ValueObject *synthetic_child;
            // We haven't made a synthetic array member for INDEX yet, so
            // lets make one and cache it for any future reference.
            synthetic_child = CreateChildAtIndex(0, true, index);

            // Cache the value if we got one back...
            if (synthetic_child)
            {
                AddSyntheticChild(index_const_str, synthetic_child);
                synthetic_child_sp = synthetic_child->GetSP();
                synthetic_child_sp->SetName(ConstString(index_str));
                synthetic_child_sp->m_is_array_item_for_pointer = true;
            }
        }
    }
    return synthetic_child_sp;
}

// This allows you to create an array member using and index
// that doesn't not fall in the normal bounds of the array.
// Many times structure can be defined as:
// struct Collection
// {
//     uint32_t item_count;
//     Item item_array[0];
// };
// The size of the "item_array" is 1, but many times in practice
// there are more items in "item_array".

ValueObjectSP
ValueObject::GetSyntheticArrayMemberFromArray (size_t index, bool can_create)
{
    ValueObjectSP synthetic_child_sp;
    if (IsArrayType ())
    {
        char index_str[64];
        snprintf(index_str, sizeof(index_str), "[%zu]", index);
        ConstString index_const_str(index_str);
        // Check if we have already created a synthetic array member in this
        // valid object. If we have we will re-use it.
        synthetic_child_sp = GetSyntheticChild (index_const_str);
        if (!synthetic_child_sp)
        {
            ValueObject *synthetic_child;
            // We haven't made a synthetic array member for INDEX yet, so
            // lets make one and cache it for any future reference.
            synthetic_child = CreateChildAtIndex(0, true, index);
            
            // Cache the value if we got one back...
            if (synthetic_child)
            {
                AddSyntheticChild(index_const_str, synthetic_child);
                synthetic_child_sp = synthetic_child->GetSP();
                synthetic_child_sp->SetName(ConstString(index_str));
                synthetic_child_sp->m_is_array_item_for_pointer = true;
            }
        }
    }
    return synthetic_child_sp;
}

ValueObjectSP
ValueObject::GetSyntheticBitFieldChild (uint32_t from, uint32_t to, bool can_create)
{
    ValueObjectSP synthetic_child_sp;
    if (IsScalarType ())
    {
        char index_str[64];
        snprintf(index_str, sizeof(index_str), "[%i-%i]", from, to);
        ConstString index_const_str(index_str);
        // Check if we have already created a synthetic array member in this
        // valid object. If we have we will re-use it.
        synthetic_child_sp = GetSyntheticChild (index_const_str);
        if (!synthetic_child_sp)
        {
            ValueObjectChild *synthetic_child;
            // We haven't made a synthetic array member for INDEX yet, so
            // lets make one and cache it for any future reference.
            synthetic_child = new ValueObjectChild(*this,
                                                      GetClangAST(),
                                                      GetClangType(),
                                                      index_const_str,
                                                      GetByteSize(),
                                                      0,
                                                      to-from+1,
                                                      from,
                                                      false,
                                                      false,
                                                      eAddressTypeInvalid);
            
            // Cache the value if we got one back...
            if (synthetic_child)
            {
                AddSyntheticChild(index_const_str, synthetic_child);
                synthetic_child_sp = synthetic_child->GetSP();
                synthetic_child_sp->SetName(ConstString(index_str));
                synthetic_child_sp->m_is_bitfield_for_scalar = true;
            }
        }
    }
    return synthetic_child_sp;
}

ValueObjectSP
ValueObject::GetSyntheticChildAtOffset(uint32_t offset, const ClangASTType& type, bool can_create)
{
    
    ValueObjectSP synthetic_child_sp;
    
    char name_str[64];
    snprintf(name_str, sizeof(name_str), "@%i", offset);
    ConstString name_const_str(name_str);
    
    // Check if we have already created a synthetic array member in this
    // valid object. If we have we will re-use it.
    synthetic_child_sp = GetSyntheticChild (name_const_str);
    
    if (synthetic_child_sp.get())
        return synthetic_child_sp;
    
    if (!can_create)
        return ValueObjectSP();
    
    ValueObjectChild *synthetic_child = new ValueObjectChild(*this,
                                                             type.GetASTContext(),
                                                             type.GetOpaqueQualType(),
                                                             name_const_str,
                                                             type.GetTypeByteSize(),
                                                             offset,
                                                             0,
                                                             0,
                                                             false,
                                                             false,
                                                             eAddressTypeInvalid);
    if (synthetic_child)
    {
        AddSyntheticChild(name_const_str, synthetic_child);
        synthetic_child_sp = synthetic_child->GetSP();
        synthetic_child_sp->SetName(name_const_str);
        synthetic_child_sp->m_is_child_at_offset = true;
    }
    return synthetic_child_sp;
}

// your expression path needs to have a leading . or ->
// (unless it somehow "looks like" an array, in which case it has
// a leading [ symbol). while the [ is meaningful and should be shown
// to the user, . and -> are just parser design, but by no means
// added information for the user.. strip them off
static const char*
SkipLeadingExpressionPathSeparators(const char* expression)
{
    if (!expression || !expression[0])
        return expression;
    if (expression[0] == '.')
        return expression+1;
    if (expression[0] == '-' && expression[1] == '>')
        return expression+2;
    return expression;
}

ValueObjectSP
ValueObject::GetSyntheticExpressionPathChild(const char* expression, bool can_create)
{
    ValueObjectSP synthetic_child_sp;
    ConstString name_const_string(expression);
    // Check if we have already created a synthetic array member in this
    // valid object. If we have we will re-use it.
    synthetic_child_sp = GetSyntheticChild (name_const_string);
    if (!synthetic_child_sp)
    {
        // We haven't made a synthetic array member for expression yet, so
        // lets make one and cache it for any future reference.
        synthetic_child_sp = GetValueForExpressionPath(expression,
                                                       NULL, NULL, NULL,
                                                       GetValueForExpressionPathOptions().DontAllowSyntheticChildren());
        
        // Cache the value if we got one back...
        if (synthetic_child_sp.get())
        {
            // FIXME: this causes a "real" child to end up with its name changed to the contents of expression
            AddSyntheticChild(name_const_string, synthetic_child_sp.get());
            synthetic_child_sp->SetName(ConstString(SkipLeadingExpressionPathSeparators(expression)));
        }
    }
    return synthetic_child_sp;
}

void
ValueObject::CalculateSyntheticValue (bool use_synthetic)
{
    if (use_synthetic == false)
        return;
    
    TargetSP target_sp(GetTargetSP());
    if (target_sp && (target_sp->GetEnableSyntheticValue() == false || target_sp->GetSuppressSyntheticValue() == true))
    {
        m_synthetic_value = NULL;
        return;
    }
    
    lldb::SyntheticChildrenSP current_synth_sp(m_synthetic_children_sp);
    
    if (!UpdateFormatsIfNeeded() && m_synthetic_value)
        return;
    
    if (m_synthetic_children_sp.get() == NULL)
        return;
    
    if (current_synth_sp == m_synthetic_children_sp && m_synthetic_value)
        return;
    
    m_synthetic_value = new ValueObjectSynthetic(*this, m_synthetic_children_sp);
}

void
ValueObject::CalculateDynamicValue (DynamicValueType use_dynamic)
{
    if (use_dynamic == eNoDynamicValues)
        return;
        
    if (!m_dynamic_value && !IsDynamic())
    {
        ExecutionContext exe_ctx (GetExecutionContextRef());
        Process *process = exe_ctx.GetProcessPtr();
        if (process && process->IsPossibleDynamicValue(*this))
        {
            ClearDynamicTypeInformation ();
            m_dynamic_value = new ValueObjectDynamicValue (*this, use_dynamic);
        }
    }
}

ValueObjectSP
ValueObject::GetDynamicValue (DynamicValueType use_dynamic)
{
    if (use_dynamic == eNoDynamicValues)
        return ValueObjectSP();
        
    if (!IsDynamic() && m_dynamic_value == NULL)
    {
        CalculateDynamicValue(use_dynamic);
    }
    if (m_dynamic_value)
        return m_dynamic_value->GetSP();
    else
        return ValueObjectSP();
}

ValueObjectSP
ValueObject::GetStaticValue()
{
    return GetSP();
}

lldb::ValueObjectSP
ValueObject::GetNonSyntheticValue ()
{
    return GetSP();
}

ValueObjectSP
ValueObject::GetSyntheticValue (bool use_synthetic)
{
    if (use_synthetic == false)
        return ValueObjectSP();

    CalculateSyntheticValue(use_synthetic);
    
    if (m_synthetic_value)
        return m_synthetic_value->GetSP();
    else
        return ValueObjectSP();
}

bool
ValueObject::HasSyntheticValue()
{
    UpdateFormatsIfNeeded();
    
    if (m_synthetic_children_sp.get() == NULL)
        return false;
    
    CalculateSyntheticValue(true);
    
    if (m_synthetic_value)
        return true;
    else
        return false;
}

bool
ValueObject::GetBaseClassPath (Stream &s)
{
    if (IsBaseClass())
    {
        bool parent_had_base_class = GetParent() && GetParent()->GetBaseClassPath (s);
        clang_type_t clang_type = GetClangType();
        std::string cxx_class_name;
        bool this_had_base_class = ClangASTContext::GetCXXClassName (clang_type, cxx_class_name);
        if (this_had_base_class)
        {
            if (parent_had_base_class)
                s.PutCString("::");
            s.PutCString(cxx_class_name.c_str());
        }
        return parent_had_base_class || this_had_base_class;
    }
    return false;
}


ValueObject *
ValueObject::GetNonBaseClassParent()
{
    if (GetParent())
    {
        if (GetParent()->IsBaseClass())
            return GetParent()->GetNonBaseClassParent();
        else
            return GetParent();
    }
    return NULL;
}

void
ValueObject::GetExpressionPath (Stream &s, bool qualify_cxx_base_classes, GetExpressionPathFormat epformat)
{
    const bool is_deref_of_parent = IsDereferenceOfParent ();

    if (is_deref_of_parent && epformat == eGetExpressionPathFormatDereferencePointers)
    {
        // this is the original format of GetExpressionPath() producing code like *(a_ptr).memberName, which is entirely
        // fine, until you put this into StackFrame::GetValueForVariableExpressionPath() which prefers to see a_ptr->memberName.
        // the eHonorPointers mode is meant to produce strings in this latter format
        s.PutCString("*(");
    }
    
    ValueObject* parent = GetParent();
    
    if (parent)
        parent->GetExpressionPath (s, qualify_cxx_base_classes, epformat);
    
    // if we are a deref_of_parent just because we are synthetic array
    // members made up to allow ptr[%d] syntax to work in variable
    // printing, then add our name ([%d]) to the expression path
    if (m_is_array_item_for_pointer && epformat == eGetExpressionPathFormatHonorPointers)
        s.PutCString(m_name.AsCString());
            
    if (!IsBaseClass())
    {
        if (!is_deref_of_parent)
        {
            ValueObject *non_base_class_parent = GetNonBaseClassParent();
            if (non_base_class_parent)
            {
                clang_type_t non_base_class_parent_clang_type = non_base_class_parent->GetClangType();
                if (non_base_class_parent_clang_type)
                {
                    const uint32_t non_base_class_parent_type_info = ClangASTContext::GetTypeInfo (non_base_class_parent_clang_type, NULL, NULL);
                    
                    if (parent && parent->IsDereferenceOfParent() && epformat == eGetExpressionPathFormatHonorPointers)
                    {
                        s.PutCString("->");
                    }
                    else
                    {                    
                        if (non_base_class_parent_type_info & ClangASTContext::eTypeIsPointer)
                        {
                            s.PutCString("->");
                        }
                        else if ((non_base_class_parent_type_info & ClangASTContext::eTypeHasChildren) &&
                                 !(non_base_class_parent_type_info & ClangASTContext::eTypeIsArray))
                        {
                            s.PutChar('.');
                        }
                    }
                }
            }

            const char *name = GetName().GetCString();
            if (name)
            {
                if (qualify_cxx_base_classes)
                {
                    if (GetBaseClassPath (s))
                        s.PutCString("::");
                }
                s.PutCString(name);
            }
        }
    }
    
    if (is_deref_of_parent && epformat == eGetExpressionPathFormatDereferencePointers)
    {
        s.PutChar(')');
    }
}

ValueObjectSP
ValueObject::GetValueForExpressionPath(const char* expression,
                                       const char** first_unparsed,
                                       ExpressionPathScanEndReason* reason_to_stop,
                                       ExpressionPathEndResultType* final_value_type,
                                       const GetValueForExpressionPathOptions& options,
                                       ExpressionPathAftermath* final_task_on_target)
{
    
    const char* dummy_first_unparsed;
    ExpressionPathScanEndReason dummy_reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnknown;
    ExpressionPathEndResultType dummy_final_value_type = ValueObject::eExpressionPathEndResultTypeInvalid;
    ExpressionPathAftermath dummy_final_task_on_target = ValueObject::eExpressionPathAftermathNothing;
    
    ValueObjectSP ret_val = GetValueForExpressionPath_Impl(expression,
                                                           first_unparsed ? first_unparsed : &dummy_first_unparsed,
                                                           reason_to_stop ? reason_to_stop : &dummy_reason_to_stop,
                                                           final_value_type ? final_value_type : &dummy_final_value_type,
                                                           options,
                                                           final_task_on_target ? final_task_on_target : &dummy_final_task_on_target);
    
    if (!final_task_on_target || *final_task_on_target == ValueObject::eExpressionPathAftermathNothing)
        return ret_val;

    if (ret_val.get() && ((final_value_type ? *final_value_type : dummy_final_value_type) == eExpressionPathEndResultTypePlain)) // I can only deref and takeaddress of plain objects
    {
        if ( (final_task_on_target ? *final_task_on_target : dummy_final_task_on_target) == ValueObject::eExpressionPathAftermathDereference)
        {
            Error error;
            ValueObjectSP final_value = ret_val->Dereference(error);
            if (error.Fail() || !final_value.get())
            {
                if (reason_to_stop)
                    *reason_to_stop = ValueObject::eExpressionPathScanEndReasonDereferencingFailed;
                if (final_value_type)
                    *final_value_type = ValueObject::eExpressionPathEndResultTypeInvalid;
                return ValueObjectSP();
            }
            else
            {
                if (final_task_on_target)
                    *final_task_on_target = ValueObject::eExpressionPathAftermathNothing;
                return final_value;
            }
        }
        if (*final_task_on_target == ValueObject::eExpressionPathAftermathTakeAddress)
        {
            Error error;
            ValueObjectSP final_value = ret_val->AddressOf(error);
            if (error.Fail() || !final_value.get())
            {
                if (reason_to_stop)
                    *reason_to_stop = ValueObject::eExpressionPathScanEndReasonTakingAddressFailed;
                if (final_value_type)
                    *final_value_type = ValueObject::eExpressionPathEndResultTypeInvalid;
                return ValueObjectSP();
            }
            else
            {
                if (final_task_on_target)
                    *final_task_on_target = ValueObject::eExpressionPathAftermathNothing;
                return final_value;
            }
        }
    }
    return ret_val; // final_task_on_target will still have its original value, so you know I did not do it
}

int
ValueObject::GetValuesForExpressionPath(const char* expression,
                                        ValueObjectListSP& list,
                                        const char** first_unparsed,
                                        ExpressionPathScanEndReason* reason_to_stop,
                                        ExpressionPathEndResultType* final_value_type,
                                        const GetValueForExpressionPathOptions& options,
                                        ExpressionPathAftermath* final_task_on_target)
{
    const char* dummy_first_unparsed;
    ExpressionPathScanEndReason dummy_reason_to_stop;
    ExpressionPathEndResultType dummy_final_value_type;
    ExpressionPathAftermath dummy_final_task_on_target = ValueObject::eExpressionPathAftermathNothing;
    
    ValueObjectSP ret_val = GetValueForExpressionPath_Impl(expression,
                                                           first_unparsed ? first_unparsed : &dummy_first_unparsed,
                                                           reason_to_stop ? reason_to_stop : &dummy_reason_to_stop,
                                                           final_value_type ? final_value_type : &dummy_final_value_type,
                                                           options,
                                                           final_task_on_target ? final_task_on_target : &dummy_final_task_on_target);
    
    if (!ret_val.get()) // if there are errors, I add nothing to the list
        return 0;
    
    if ( (reason_to_stop ? *reason_to_stop : dummy_reason_to_stop) != eExpressionPathScanEndReasonArrayRangeOperatorMet)
    {
        // I need not expand a range, just post-process the final value and return
        if (!final_task_on_target || *final_task_on_target == ValueObject::eExpressionPathAftermathNothing)
        {
            list->Append(ret_val);
            return 1;
        }
        if (ret_val.get() && (final_value_type ? *final_value_type : dummy_final_value_type) == eExpressionPathEndResultTypePlain) // I can only deref and takeaddress of plain objects
        {
            if (*final_task_on_target == ValueObject::eExpressionPathAftermathDereference)
            {
                Error error;
                ValueObjectSP final_value = ret_val->Dereference(error);
                if (error.Fail() || !final_value.get())
                {
                    if (reason_to_stop)
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonDereferencingFailed;
                    if (final_value_type)
                        *final_value_type = ValueObject::eExpressionPathEndResultTypeInvalid;
                    return 0;
                }
                else
                {
                    *final_task_on_target = ValueObject::eExpressionPathAftermathNothing;
                    list->Append(final_value);
                    return 1;
                }
            }
            if (*final_task_on_target == ValueObject::eExpressionPathAftermathTakeAddress)
            {
                Error error;
                ValueObjectSP final_value = ret_val->AddressOf(error);
                if (error.Fail() || !final_value.get())
                {
                    if (reason_to_stop)
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonTakingAddressFailed;
                    if (final_value_type)
                        *final_value_type = ValueObject::eExpressionPathEndResultTypeInvalid;
                    return 0;
                }
                else
                {
                    *final_task_on_target = ValueObject::eExpressionPathAftermathNothing;
                    list->Append(final_value);
                    return 1;
                }
            }
        }
    }
    else
    {
        return ExpandArraySliceExpression(first_unparsed ? *first_unparsed : dummy_first_unparsed,
                                          first_unparsed ? first_unparsed : &dummy_first_unparsed,
                                          ret_val,
                                          list,
                                          reason_to_stop ? reason_to_stop : &dummy_reason_to_stop,
                                          final_value_type ? final_value_type : &dummy_final_value_type,
                                          options,
                                          final_task_on_target ? final_task_on_target : &dummy_final_task_on_target);
    }
    // in any non-covered case, just do the obviously right thing
    list->Append(ret_val);
    return 1;
}

ValueObjectSP
ValueObject::GetValueForExpressionPath_Impl(const char* expression_cstr,
                                            const char** first_unparsed,
                                            ExpressionPathScanEndReason* reason_to_stop,
                                            ExpressionPathEndResultType* final_result,
                                            const GetValueForExpressionPathOptions& options,
                                            ExpressionPathAftermath* what_next)
{
    ValueObjectSP root = GetSP();
    
    if (!root.get())
        return ValueObjectSP();
    
    *first_unparsed = expression_cstr;
    
    while (true)
    {
        
        const char* expression_cstr = *first_unparsed; // hide the top level expression_cstr
        
        clang_type_t root_clang_type = root->GetClangType();
        clang_type_t pointee_clang_type;
        Flags root_clang_type_info,pointee_clang_type_info;
        
        root_clang_type_info = Flags(ClangASTContext::GetTypeInfo(root_clang_type, GetClangAST(), &pointee_clang_type));
        if (pointee_clang_type)
            pointee_clang_type_info = Flags(ClangASTContext::GetTypeInfo(pointee_clang_type, GetClangAST(), NULL));
        
        if (!expression_cstr || *expression_cstr == '\0')
        {
            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonEndOfString;
            return root;
        }
        
        switch (*expression_cstr)
        {
            case '-':
            {
                if (options.m_check_dot_vs_arrow_syntax &&
                    root_clang_type_info.Test(ClangASTContext::eTypeIsPointer) ) // if you are trying to use -> on a non-pointer and I must catch the error
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eExpressionPathScanEndReasonArrowInsteadOfDot;
                    *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                    return ValueObjectSP();
                }
                if (root_clang_type_info.Test(ClangASTContext::eTypeIsObjC) &&  // if yo are trying to extract an ObjC IVar when this is forbidden
                    root_clang_type_info.Test(ClangASTContext::eTypeIsPointer) &&
                    options.m_no_fragile_ivar)
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eExpressionPathScanEndReasonFragileIVarNotAllowed;
                    *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                    return ValueObjectSP();
                }
                if (expression_cstr[1] != '>')
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                    *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                    return ValueObjectSP();
                }
                expression_cstr++; // skip the -
            }
            case '.': // or fallthrough from ->
            {
                if (options.m_check_dot_vs_arrow_syntax && *expression_cstr == '.' &&
                    root_clang_type_info.Test(ClangASTContext::eTypeIsPointer)) // if you are trying to use . on a pointer and I must catch the error
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eExpressionPathScanEndReasonDotInsteadOfArrow;
                    *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                    return ValueObjectSP();
                }
                expression_cstr++; // skip .
                const char *next_separator = strpbrk(expression_cstr+1,"-.[");
                ConstString child_name;
                if (!next_separator) // if no other separator just expand this last layer
                {
                    child_name.SetCString (expression_cstr);
                    ValueObjectSP child_valobj_sp = root->GetChildMemberWithName(child_name, true);
                    
                    if (child_valobj_sp.get()) // we know we are done, so just return
                    {
                        *first_unparsed = "";
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonEndOfString;
                        *final_result = ValueObject::eExpressionPathEndResultTypePlain;
                        return child_valobj_sp;
                    }
                    else if (options.m_no_synthetic_children == false) // let's try with synthetic children
                    {
                        if (root->IsSynthetic())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }

                        child_valobj_sp = root->GetSyntheticValue();
                        if (child_valobj_sp.get())
                            child_valobj_sp = child_valobj_sp->GetChildMemberWithName(child_name, true);
                    }
                    
                    // if we are here and options.m_no_synthetic_children is true, child_valobj_sp is going to be a NULL SP,
                    // so we hit the "else" branch, and return an error
                    if(child_valobj_sp.get()) // if it worked, just return
                    {
                        *first_unparsed = "";
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonEndOfString;
                        *final_result = ValueObject::eExpressionPathEndResultTypePlain;
                        return child_valobj_sp;
                    }
                    else
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return ValueObjectSP();
                    }
                }
                else // other layers do expand
                {
                    child_name.SetCStringWithLength(expression_cstr, next_separator - expression_cstr);
                    ValueObjectSP child_valobj_sp = root->GetChildMemberWithName(child_name, true);
                    if (child_valobj_sp.get()) // store the new root and move on
                    {
                        root = child_valobj_sp;
                        *first_unparsed = next_separator;
                        *final_result = ValueObject::eExpressionPathEndResultTypePlain;
                        continue;
                    }
                    else if (options.m_no_synthetic_children == false) // let's try with synthetic children
                    {
                        if (root->IsSynthetic())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }
                        
                        child_valobj_sp = root->GetSyntheticValue(true);
                        if (child_valobj_sp)
                            child_valobj_sp = child_valobj_sp->GetChildMemberWithName(child_name, true);
                    }
                    
                    // if we are here and options.m_no_synthetic_children is true, child_valobj_sp is going to be a NULL SP,
                    // so we hit the "else" branch, and return an error
                    if(child_valobj_sp.get()) // if it worked, move on
                    {
                        root = child_valobj_sp;
                        *first_unparsed = next_separator;
                        *final_result = ValueObject::eExpressionPathEndResultTypePlain;
                        continue;
                    }
                    else
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return ValueObjectSP();
                    }
                }
                break;
            }
            case '[':
            {
                if (!root_clang_type_info.Test(ClangASTContext::eTypeIsArray) && !root_clang_type_info.Test(ClangASTContext::eTypeIsPointer)) // if this is not a T[] nor a T*
                {
                    if (!root_clang_type_info.Test(ClangASTContext::eTypeIsScalar)) // if this is not even a scalar...
                    {
                        if (options.m_no_synthetic_children) // ...only chance left is synthetic
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorInvalid;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }
                    }
                    else if (!options.m_allow_bitfields_syntax) // if this is a scalar, check that we can expand bitfields
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorNotAllowed;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return ValueObjectSP();
                    }
                }
                if (*(expression_cstr+1) == ']') // if this is an unbounded range it only works for arrays
                {
                    if (!root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonEmptyRangeNotAllowed;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return ValueObjectSP();
                    }
                    else // even if something follows, we cannot expand unbounded ranges, just let the caller do it
                    {
                        *first_unparsed = expression_cstr+2;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonArrayRangeOperatorMet;
                        *final_result = ValueObject::eExpressionPathEndResultTypeUnboundedRange;
                        return root;
                    }
                }
                const char *separator_position = ::strchr(expression_cstr+1,'-');
                const char *close_bracket_position = ::strchr(expression_cstr+1,']');
                if (!close_bracket_position) // if there is no ], this is a syntax error
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                    *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                    return ValueObjectSP();
                }
                if (!separator_position || separator_position > close_bracket_position) // if no separator, this is either [] or [N]
                {
                    char *end = NULL;
                    unsigned long index = ::strtoul (expression_cstr+1, &end, 0);
                    if (!end || end != close_bracket_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return ValueObjectSP();
                    }
                    if (end - expression_cstr == 1) // if this is [], only return a valid value for arrays
                    {
                        if (root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                        {
                            *first_unparsed = expression_cstr+2;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonArrayRangeOperatorMet;
                            *final_result = ValueObject::eExpressionPathEndResultTypeUnboundedRange;
                            return root;
                        }
                        else
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonEmptyRangeNotAllowed;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }
                    }
                    // from here on we do have a valid index
                    if (root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                    {
                        ValueObjectSP child_valobj_sp = root->GetChildAtIndex(index, true);
                        if (!child_valobj_sp)
                            child_valobj_sp = root->GetSyntheticArrayMemberFromArray(index, true);
                        if (!child_valobj_sp)
                            if (root->HasSyntheticValue() && root->GetSyntheticValue()->GetNumChildren() > index)
                                child_valobj_sp = root->GetSyntheticValue()->GetChildAtIndex(index, true);
                        if (child_valobj_sp)
                        {
                            root = child_valobj_sp;
                            *first_unparsed = end+1; // skip ]
                            *final_result = ValueObject::eExpressionPathEndResultTypePlain;
                            continue;
                        }
                        else
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }
                    }
                    else if (root_clang_type_info.Test(ClangASTContext::eTypeIsPointer))
                    {
                        if (*what_next == ValueObject::eExpressionPathAftermathDereference &&  // if this is a ptr-to-scalar, I am accessing it by index and I would have deref'ed anyway, then do it now and use this as a bitfield
                            pointee_clang_type_info.Test(ClangASTContext::eTypeIsScalar))
                        {
                            Error error;
                            root = root->Dereference(error);
                            if (error.Fail() || !root.get())
                            {
                                *first_unparsed = expression_cstr;
                                *reason_to_stop = ValueObject::eExpressionPathScanEndReasonDereferencingFailed;
                                *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                                return ValueObjectSP();
                            }
                            else
                            {
                                *what_next = eExpressionPathAftermathNothing;
                                continue;
                            }
                        }
                        else
                        {
                            if (ClangASTType::GetMinimumLanguage(root->GetClangAST(),
                                                                 root->GetClangType()) == eLanguageTypeObjC
                                && ClangASTContext::IsPointerType(ClangASTType::GetPointeeType(root->GetClangType())) == false
                                && root->HasSyntheticValue()
                                && options.m_no_synthetic_children == false)
                            {
                                root = root->GetSyntheticValue()->GetChildAtIndex(index, true);
                            }
                            else
                                root = root->GetSyntheticArrayMemberFromPointer(index, true);
                            if (!root.get())
                            {
                                *first_unparsed = expression_cstr;
                                *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                                *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                                return ValueObjectSP();
                            }
                            else
                            {
                                *first_unparsed = end+1; // skip ]
                                *final_result = ValueObject::eExpressionPathEndResultTypePlain;
                                continue;
                            }
                        }
                    }
                    else if (ClangASTContext::IsScalarType(root_clang_type))
                    {
                        root = root->GetSyntheticBitFieldChild(index, index, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }
                        else // we do not know how to expand members of bitfields, so we just return and let the caller do any further processing
                        {
                            *first_unparsed = end+1; // skip ]
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonBitfieldRangeOperatorMet;
                            *final_result = ValueObject::eExpressionPathEndResultTypeBitfield;
                            return root;
                        }
                    }
                    else if (options.m_no_synthetic_children == false)
                    {
                        if (root->HasSyntheticValue())
                            root = root->GetSyntheticValue();
                        else if (!root->IsSynthetic())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonSyntheticValueMissing;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }
                        // if we are here, then root itself is a synthetic VO.. should be good to go
                        
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonSyntheticValueMissing;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }
                        root = root->GetChildAtIndex(index, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }
                        else
                        {
                            *first_unparsed = end+1; // skip ]
                            *final_result = ValueObject::eExpressionPathEndResultTypePlain;
                            continue;
                        }
                    }
                    else
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return ValueObjectSP();
                    }
                }
                else // we have a low and a high index
                {
                    char *end = NULL;
                    unsigned long index_lower = ::strtoul (expression_cstr+1, &end, 0);
                    if (!end || end != separator_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return ValueObjectSP();
                    }
                    unsigned long index_higher = ::strtoul (separator_position+1, &end, 0);
                    if (!end || end != close_bracket_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return ValueObjectSP();
                    }
                    if (index_lower > index_higher) // swap indices if required
                    {
                        unsigned long temp = index_lower;
                        index_lower = index_higher;
                        index_higher = temp;
                    }
                    if (root_clang_type_info.Test(ClangASTContext::eTypeIsScalar)) // expansion only works for scalars
                    {
                        root = root->GetSyntheticBitFieldChild(index_lower, index_higher, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }
                        else
                        {
                            *first_unparsed = end+1; // skip ]
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonBitfieldRangeOperatorMet;
                            *final_result = ValueObject::eExpressionPathEndResultTypeBitfield;
                            return root;
                        }
                    }
                    else if (root_clang_type_info.Test(ClangASTContext::eTypeIsPointer) && // if this is a ptr-to-scalar, I am accessing it by index and I would have deref'ed anyway, then do it now and use this as a bitfield
                             *what_next == ValueObject::eExpressionPathAftermathDereference &&
                             pointee_clang_type_info.Test(ClangASTContext::eTypeIsScalar))
                    {
                        Error error;
                        root = root->Dereference(error);
                        if (error.Fail() || !root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonDereferencingFailed;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return ValueObjectSP();
                        }
                        else
                        {
                            *what_next = ValueObject::eExpressionPathAftermathNothing;
                            continue;
                        }
                    }
                    else
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonArrayRangeOperatorMet;
                        *final_result = ValueObject::eExpressionPathEndResultTypeBoundedRange;
                        return root;
                    }
                }
                break;
            }
            default: // some non-separator is in the way
            {
                *first_unparsed = expression_cstr;
                *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                return ValueObjectSP();
                break;
            }
        }
    }
}

int
ValueObject::ExpandArraySliceExpression(const char* expression_cstr,
                                        const char** first_unparsed,
                                        ValueObjectSP root,
                                        ValueObjectListSP& list,
                                        ExpressionPathScanEndReason* reason_to_stop,
                                        ExpressionPathEndResultType* final_result,
                                        const GetValueForExpressionPathOptions& options,
                                        ExpressionPathAftermath* what_next)
{
    if (!root.get())
        return 0;
    
    *first_unparsed = expression_cstr;
    
    while (true)
    {
        
        const char* expression_cstr = *first_unparsed; // hide the top level expression_cstr
        
        clang_type_t root_clang_type = root->GetClangType();
        clang_type_t pointee_clang_type;
        Flags root_clang_type_info,pointee_clang_type_info;
        
        root_clang_type_info = Flags(ClangASTContext::GetTypeInfo(root_clang_type, GetClangAST(), &pointee_clang_type));
        if (pointee_clang_type)
            pointee_clang_type_info = Flags(ClangASTContext::GetTypeInfo(pointee_clang_type, GetClangAST(), NULL));
        
        if (!expression_cstr || *expression_cstr == '\0')
        {
            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonEndOfString;
            list->Append(root);
            return 1;
        }
        
        switch (*expression_cstr)
        {
            case '[':
            {
                if (!root_clang_type_info.Test(ClangASTContext::eTypeIsArray) && !root_clang_type_info.Test(ClangASTContext::eTypeIsPointer)) // if this is not a T[] nor a T*
                {
                    if (!root_clang_type_info.Test(ClangASTContext::eTypeIsScalar)) // if this is not even a scalar, this syntax is just plain wrong!
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorInvalid;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return 0;
                    }
                    else if (!options.m_allow_bitfields_syntax) // if this is a scalar, check that we can expand bitfields
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorNotAllowed;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return 0;
                    }
                }
                if (*(expression_cstr+1) == ']') // if this is an unbounded range it only works for arrays
                {
                    if (!root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonEmptyRangeNotAllowed;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return 0;
                    }
                    else // expand this into list
                    {
                        const size_t max_index = root->GetNumChildren() - 1;
                        for (size_t index = 0; index < max_index; index++)
                        {
                            ValueObjectSP child = 
                                root->GetChildAtIndex(index, true);
                            list->Append(child);
                        }
                        *first_unparsed = expression_cstr+2;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorExpanded;
                        *final_result = ValueObject::eExpressionPathEndResultTypeValueObjectList;
                        return max_index; // tell me number of items I added to the VOList
                    }
                }
                const char *separator_position = ::strchr(expression_cstr+1,'-');
                const char *close_bracket_position = ::strchr(expression_cstr+1,']');
                if (!close_bracket_position) // if there is no ], this is a syntax error
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                    *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                    return 0;
                }
                if (!separator_position || separator_position > close_bracket_position) // if no separator, this is either [] or [N]
                {
                    char *end = NULL;
                    unsigned long index = ::strtoul (expression_cstr+1, &end, 0);
                    if (!end || end != close_bracket_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return 0;
                    }
                    if (end - expression_cstr == 1) // if this is [], only return a valid value for arrays
                    {
                        if (root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                        {
                            const size_t max_index = root->GetNumChildren() - 1;
                            for (size_t index = 0; index < max_index; index++)
                            {
                                ValueObjectSP child = 
                                root->GetChildAtIndex(index, true);
                                list->Append(child);
                            }
                            *first_unparsed = expression_cstr+2;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorExpanded;
                            *final_result = ValueObject::eExpressionPathEndResultTypeValueObjectList;
                            return max_index; // tell me number of items I added to the VOList
                        }
                        else
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonEmptyRangeNotAllowed;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return 0;
                        }
                    }
                    // from here on we do have a valid index
                    if (root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                    {
                        root = root->GetChildAtIndex(index, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return 0;
                        }
                        else
                        {
                            list->Append(root);
                            *first_unparsed = end+1; // skip ]
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorExpanded;
                            *final_result = ValueObject::eExpressionPathEndResultTypeValueObjectList;
                            return 1;
                        }
                    }
                    else if (root_clang_type_info.Test(ClangASTContext::eTypeIsPointer))
                    {
                        if (*what_next == ValueObject::eExpressionPathAftermathDereference &&  // if this is a ptr-to-scalar, I am accessing it by index and I would have deref'ed anyway, then do it now and use this as a bitfield
                            pointee_clang_type_info.Test(ClangASTContext::eTypeIsScalar))
                        {
                            Error error;
                            root = root->Dereference(error);
                            if (error.Fail() || !root.get())
                            {
                                *first_unparsed = expression_cstr;
                                *reason_to_stop = ValueObject::eExpressionPathScanEndReasonDereferencingFailed;
                                *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                                return 0;
                            }
                            else
                            {
                                *what_next = eExpressionPathAftermathNothing;
                                continue;
                            }
                        }
                        else
                        {
                            root = root->GetSyntheticArrayMemberFromPointer(index, true);
                            if (!root.get())
                            {
                                *first_unparsed = expression_cstr;
                                *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                                *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                                return 0;
                            }
                            else
                            {
                                list->Append(root);
                                *first_unparsed = end+1; // skip ]
                                *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorExpanded;
                                *final_result = ValueObject::eExpressionPathEndResultTypeValueObjectList;
                                return 1;
                            }
                        }
                    }
                    else /*if (ClangASTContext::IsScalarType(root_clang_type))*/
                    {
                        root = root->GetSyntheticBitFieldChild(index, index, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return 0;
                        }
                        else // we do not know how to expand members of bitfields, so we just return and let the caller do any further processing
                        {
                            list->Append(root);
                            *first_unparsed = end+1; // skip ]
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorExpanded;
                            *final_result = ValueObject::eExpressionPathEndResultTypeValueObjectList;
                            return 1;
                        }
                    }
                }
                else // we have a low and a high index
                {
                    char *end = NULL;
                    unsigned long index_lower = ::strtoul (expression_cstr+1, &end, 0);
                    if (!end || end != separator_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return 0;
                    }
                    unsigned long index_higher = ::strtoul (separator_position+1, &end, 0);
                    if (!end || end != close_bracket_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                        *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                        return 0;
                    }
                    if (index_lower > index_higher) // swap indices if required
                    {
                        unsigned long temp = index_lower;
                        index_lower = index_higher;
                        index_higher = temp;
                    }
                    if (root_clang_type_info.Test(ClangASTContext::eTypeIsScalar)) // expansion only works for scalars
                    {
                        root = root->GetSyntheticBitFieldChild(index_lower, index_higher, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonNoSuchChild;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return 0;
                        }
                        else
                        {
                            list->Append(root);
                            *first_unparsed = end+1; // skip ]
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorExpanded;
                            *final_result = ValueObject::eExpressionPathEndResultTypeValueObjectList;
                            return 1;
                        }
                    }
                    else if (root_clang_type_info.Test(ClangASTContext::eTypeIsPointer) && // if this is a ptr-to-scalar, I am accessing it by index and I would have deref'ed anyway, then do it now and use this as a bitfield
                             *what_next == ValueObject::eExpressionPathAftermathDereference &&
                             pointee_clang_type_info.Test(ClangASTContext::eTypeIsScalar))
                    {
                        Error error;
                        root = root->Dereference(error);
                        if (error.Fail() || !root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eExpressionPathScanEndReasonDereferencingFailed;
                            *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                            return 0;
                        }
                        else
                        {
                            *what_next = ValueObject::eExpressionPathAftermathNothing;
                            continue;
                        }
                    }
                    else
                    {
                        for (unsigned long index = index_lower;
                             index <= index_higher; index++)
                        {
                            ValueObjectSP child = 
                                root->GetChildAtIndex(index, true);
                            list->Append(child);
                        }
                        *first_unparsed = end+1;
                        *reason_to_stop = ValueObject::eExpressionPathScanEndReasonRangeOperatorExpanded;
                        *final_result = ValueObject::eExpressionPathEndResultTypeValueObjectList;
                        return index_higher-index_lower+1; // tell me number of items I added to the VOList
                    }
                }
                break;
            }
            default: // some non-[ separator, or something entirely wrong, is in the way
            {
                *first_unparsed = expression_cstr;
                *reason_to_stop = ValueObject::eExpressionPathScanEndReasonUnexpectedSymbol;
                *final_result = ValueObject::eExpressionPathEndResultTypeInvalid;
                return 0;
                break;
            }
        }
    }
}

static void
DumpValueObject_Impl (Stream &s,
                      ValueObject *valobj,
                      const ValueObject::DumpValueObjectOptions& options,
                      uint32_t ptr_depth,
                      uint32_t curr_depth)
{
    if (valobj)
    {
        bool update_success = valobj->UpdateValueIfNeeded (true);

        const char *root_valobj_name = 
            options.m_root_valobj_name.empty() ? 
                valobj->GetName().AsCString() :
                options.m_root_valobj_name.c_str();
        
        if (update_success && options.m_use_dynamic != eNoDynamicValues)
        {
            ValueObject *dynamic_value = valobj->GetDynamicValue(options.m_use_dynamic).get();
            if (dynamic_value)
                valobj = dynamic_value;
        }
        
        clang_type_t clang_type = valobj->GetClangType();

        const Flags type_flags (ClangASTContext::GetTypeInfo (clang_type, NULL, NULL));
        const char *err_cstr = NULL;
        const bool has_children = type_flags.Test (ClangASTContext::eTypeHasChildren);
        const bool has_value = type_flags.Test (ClangASTContext::eTypeHasValue);
        
        const bool print_valobj = options.m_flat_output == false || has_value;
        
        if (print_valobj)
        {
            if (options.m_show_location)
            {
                s.Printf("%s: ", valobj->GetLocationAsCString());
            }

            s.Indent();
            
            bool show_type = true;
            // if we are at the root-level and been asked to hide the root's type, then hide it
            if (curr_depth == 0 && options.m_hide_root_type)
                show_type = false;
            else
            // otherwise decide according to the usual rules (asked to show types - always at the root level)
                show_type = options.m_show_types || (curr_depth == 0 && !options.m_flat_output);
            
            if (show_type)
            {
                // Some ValueObjects don't have types (like registers sets). Only print
                // the type if there is one to print
                ConstString qualified_type_name(valobj->GetQualifiedTypeName());
                if (qualified_type_name)
                    s.Printf("(%s) ", qualified_type_name.GetCString());
            }

            if (options.m_flat_output)
            {
                // If we are showing types, also qualify the C++ base classes 
                const bool qualify_cxx_base_classes = options.m_show_types;
                if (!options.m_hide_name)
                {
                    valobj->GetExpressionPath(s, qualify_cxx_base_classes);
                    s.PutCString(" =");
                }
            }
            else if (!options.m_hide_name)
            {
                const char *name_cstr = root_valobj_name ? root_valobj_name : valobj->GetName().AsCString("");
                s.Printf ("%s =", name_cstr);
            }

            if (!options.m_scope_already_checked && !valobj->IsInScope())
            {
                err_cstr = "out of scope";
            }
        }
        
        std::string summary_str;
        std::string value_str;
        const char *val_cstr = NULL;
        const char *sum_cstr = NULL;
        TypeSummaryImpl* entry = options.m_summary_sp ? options.m_summary_sp.get() : valobj->GetSummaryFormat().get();
        
        if (options.m_omit_summary_depth > 0)
            entry = NULL;
        
        bool is_nil = valobj->IsObjCNil();
        
        if (err_cstr == NULL)
        {
            if (options.m_format != eFormatDefault && options.m_format != valobj->GetFormat())
            {
                valobj->GetValueAsCString(options.m_format,
                                          value_str);
            }
            else
            {
                val_cstr = valobj->GetValueAsCString();
                if (val_cstr)
                    value_str = val_cstr;
            }
            err_cstr = valobj->GetError().AsCString();
        }

        if (err_cstr)
        {
            s.Printf (" <%s>\n", err_cstr);
        }
        else
        {
            const bool is_ref = type_flags.Test (ClangASTContext::eTypeIsReference);
            if (print_valobj)
            {
                if (is_nil)
                    sum_cstr = "nil";
                else if (options.m_omit_summary_depth == 0)
                {
                    if (options.m_summary_sp)
                    {
                        valobj->GetSummaryAsCString(entry, summary_str);
                        sum_cstr = summary_str.c_str();
                    }
                    else
                        sum_cstr = valobj->GetSummaryAsCString();
                }

                // Make sure we have a value and make sure the summary didn't
                // specify that the value should not be printed - and do not print
                // the value if this thing is nil
                // (but show the value if the user passes a format explicitly)
                if (!is_nil && !value_str.empty() && (entry == NULL || (entry->DoesPrintValue() || options.m_format != eFormatDefault) || sum_cstr == NULL) && !options.m_hide_value)
                    s.Printf(" %s", value_str.c_str());

                if (sum_cstr)
                    s.Printf(" %s", sum_cstr);
                
                // let's avoid the overly verbose no description error for a nil thing
                if (options.m_use_objc && !is_nil)
                {
                    if (!options.m_hide_value || !options.m_hide_name)
                        s.Printf(" ");
                    const char *object_desc = valobj->GetObjectDescription();
                    if (object_desc)
                        s.Printf("%s\n", object_desc);
                    else
                        s.Printf ("[no Objective-C description available]\n");
                    return;
                }
            }

            if (curr_depth < options.m_max_depth)
            {
                // We will show children for all concrete types. We won't show
                // pointer contents unless a pointer depth has been specified.
                // We won't reference contents unless the reference is the 
                // root object (depth of zero).
                bool print_children = true;

                // Use a new temporary pointer depth in case we override the
                // current pointer depth below...
                uint32_t curr_ptr_depth = ptr_depth;

                const bool is_ptr = type_flags.Test (ClangASTContext::eTypeIsPointer);
                if (is_ptr || is_ref)
                {
                    // We have a pointer or reference whose value is an address.
                    // Make sure that address is not NULL
                    AddressType ptr_address_type;
                    if (valobj->GetPointerValue (&ptr_address_type) == 0)
                        print_children = false;

                    else if (is_ref && curr_depth == 0)
                    {
                        // If this is the root object (depth is zero) that we are showing
                        // and it is a reference, and no pointer depth has been supplied
                        // print out what it references. Don't do this at deeper depths
                        // otherwise we can end up with infinite recursion...
                        curr_ptr_depth = 1;
                    }
                    
                    if (curr_ptr_depth == 0)
                        print_children = false;
                }
                
                if (print_children && (!entry || entry->DoesPrintChildren() || !sum_cstr))
                {
                    ValueObjectSP synth_valobj_sp = valobj->GetSyntheticValue (options.m_use_synthetic);
                    ValueObject* synth_valobj = (synth_valobj_sp ? synth_valobj_sp.get() : valobj);
                    
                    size_t num_children = synth_valobj->GetNumChildren();
                    bool print_dotdotdot = false;
                    if (num_children)
                    {
                        if (options.m_flat_output)
                        {
                            if (print_valobj)
                                s.EOL();
                        }
                        else
                        {
                            if (print_valobj)
                                s.PutCString(is_ref ? ": {\n" : " {\n");
                            s.IndentMore();
                        }
                        
                        const size_t max_num_children = valobj->GetTargetSP()->GetMaximumNumberOfChildrenToDisplay();
                        
                        if (num_children > max_num_children && !options.m_ignore_cap)
                        {
                            num_children = max_num_children;
                            print_dotdotdot = true;
                        }

                        ValueObject::DumpValueObjectOptions child_options(options);
                        child_options.SetFormat(options.m_format).SetSummary().SetRootValueObjectName();
                        child_options.SetScopeChecked(true).SetHideName(options.m_hide_name).SetHideValue(options.m_hide_value)
                        .SetOmitSummaryDepth(child_options.m_omit_summary_depth > 1 ? child_options.m_omit_summary_depth - 1 : 0);
                        for (size_t idx=0; idx<num_children; ++idx)
                        {
                            ValueObjectSP child_sp(synth_valobj->GetChildAtIndex(idx, true));
                            if (child_sp.get())
                            {
                                DumpValueObject_Impl (s,
                                                      child_sp.get(),
                                                      child_options,
                                                      (is_ptr || is_ref) ? curr_ptr_depth - 1 : curr_ptr_depth,
                                                      curr_depth + 1);
                            }
                        }

                        if (!options.m_flat_output)
                        {
                            if (print_dotdotdot)
                            {
                                ExecutionContext exe_ctx (valobj->GetExecutionContextRef());
                                Target *target = exe_ctx.GetTargetPtr();
                                if (target)
                                    target->GetDebugger().GetCommandInterpreter().ChildrenTruncated();
                                s.Indent("...\n");
                            }
                            s.IndentLess();
                            s.Indent("}\n");
                        }
                    }
                    else if (has_children)
                    {
                        // Aggregate, no children...
                        if (print_valobj)
                            s.PutCString(" {}\n");
                    }
                    else
                    {
                        if (print_valobj)
                            s.EOL();
                    }

                }
                else
                {  
                    s.EOL();
                }
            }
            else
            {
                if (has_children && print_valobj)
                {
                    s.PutCString("{...}\n");
                }
            }
        }
    }
}

void
ValueObject::LogValueObject (Log *log,
                             ValueObject *valobj)
{
    if (log && valobj)
        return LogValueObject (log, valobj, DumpValueObjectOptions::DefaultOptions());
}

void
ValueObject::LogValueObject (Log *log,
                             ValueObject *valobj,
                             const DumpValueObjectOptions& options)
{
    if (log && valobj)
    {
        StreamString s;
        ValueObject::DumpValueObject (s, valobj, options);
        if (s.GetSize())
            log->PutCString(s.GetData());
    }
}

void
ValueObject::DumpValueObject (Stream &s,
                              ValueObject *valobj)
{
    
    if (!valobj)
        return;
    
    DumpValueObject_Impl(s,
                         valobj,
                         DumpValueObjectOptions::DefaultOptions(),
                         0,
                         0);
}

void
ValueObject::DumpValueObject (Stream &s,
                              ValueObject *valobj,
                              const DumpValueObjectOptions& options)
{
    DumpValueObject_Impl(s,
                         valobj,
                         options,
                         options.m_max_ptr_depth, // max pointer depth allowed, we will go down from here
                         0 // current object depth is 0 since we are just starting
                         );
}

ValueObjectSP
ValueObject::CreateConstantValue (const ConstString &name)
{
    ValueObjectSP valobj_sp;
    
    if (UpdateValueIfNeeded(false) && m_error.Success())
    {
        ExecutionContext exe_ctx (GetExecutionContextRef());
        clang::ASTContext *ast = GetClangAST ();
        
        DataExtractor data;
        data.SetByteOrder (m_data.GetByteOrder());
        data.SetAddressByteSize(m_data.GetAddressByteSize());
        
        if (IsBitfield())
        {
            Value v(Scalar(GetValueAsUnsigned(UINT64_MAX)));
            m_error = v.GetValueAsData (&exe_ctx, ast, data, 0, GetModule().get());
        }
        else
            m_error = m_value.GetValueAsData (&exe_ctx, ast, data, 0, GetModule().get());
        
        valobj_sp = ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(), 
                                                    ast,
                                                    GetClangType(),
                                                    name,
                                                    data,
                                                    GetAddressOf());
    }
    
    if (!valobj_sp)
    {
        valobj_sp = ValueObjectConstResult::Create (NULL, m_error);
    }
    return valobj_sp;
}

ValueObjectSP
ValueObject::Dereference (Error &error)
{
    if (m_deref_valobj)
        return m_deref_valobj->GetSP();
        
    const bool is_pointer_type = IsPointerType();
    if (is_pointer_type)
    {
        bool omit_empty_base_classes = true;
        bool ignore_array_bounds = false;

        std::string child_name_str;
        uint32_t child_byte_size = 0;
        int32_t child_byte_offset = 0;
        uint32_t child_bitfield_bit_size = 0;
        uint32_t child_bitfield_bit_offset = 0;
        bool child_is_base_class = false;
        bool child_is_deref_of_parent = false;
        const bool transparent_pointers = false;
        clang::ASTContext *clang_ast = GetClangAST();
        clang_type_t clang_type = GetClangType();
        clang_type_t child_clang_type;

        ExecutionContext exe_ctx (GetExecutionContextRef());
        
        child_clang_type = ClangASTContext::GetChildClangTypeAtIndex (&exe_ctx,
                                                                      clang_ast,
                                                                      GetName().GetCString(),
                                                                      clang_type,
                                                                      0,
                                                                      transparent_pointers,
                                                                      omit_empty_base_classes,
                                                                      ignore_array_bounds,
                                                                      child_name_str,
                                                                      child_byte_size,
                                                                      child_byte_offset,
                                                                      child_bitfield_bit_size,
                                                                      child_bitfield_bit_offset,
                                                                      child_is_base_class,
                                                                      child_is_deref_of_parent);
        if (child_clang_type && child_byte_size)
        {
            ConstString child_name;
            if (!child_name_str.empty())
                child_name.SetCString (child_name_str.c_str());

            m_deref_valobj = new ValueObjectChild (*this,
                                                   clang_ast,
                                                   child_clang_type,
                                                   child_name,
                                                   child_byte_size,
                                                   child_byte_offset,
                                                   child_bitfield_bit_size,
                                                   child_bitfield_bit_offset,
                                                   child_is_base_class,
                                                   child_is_deref_of_parent,
                                                   eAddressTypeInvalid);
        }
    }

    if (m_deref_valobj)
    {
        error.Clear();
        return m_deref_valobj->GetSP();
    }
    else
    {
        StreamString strm;
        GetExpressionPath(strm, true);

        if (is_pointer_type)
            error.SetErrorStringWithFormat("dereference failed: (%s) %s", GetTypeName().AsCString("<invalid type>"), strm.GetString().c_str());
        else
            error.SetErrorStringWithFormat("not a pointer type: (%s) %s", GetTypeName().AsCString("<invalid type>"), strm.GetString().c_str());
        return ValueObjectSP();
    }
}

ValueObjectSP
ValueObject::AddressOf (Error &error)
{
    if (m_addr_of_valobj_sp)
        return m_addr_of_valobj_sp;
        
    AddressType address_type = eAddressTypeInvalid;
    const bool scalar_is_load_address = false;
    addr_t addr = GetAddressOf (scalar_is_load_address, &address_type);
    error.Clear();
    if (addr != LLDB_INVALID_ADDRESS)
    {
        switch (address_type)
        {
        case eAddressTypeInvalid:
            {
                StreamString expr_path_strm;
                GetExpressionPath(expr_path_strm, true);
                error.SetErrorStringWithFormat("'%s' is not in memory", expr_path_strm.GetString().c_str());
            }
            break;

        case eAddressTypeFile:
        case eAddressTypeLoad:
        case eAddressTypeHost:
            {
                clang::ASTContext *ast = GetClangAST();
                clang_type_t clang_type = GetClangType();
                if (ast && clang_type)
                {
                    std::string name (1, '&');
                    name.append (m_name.AsCString(""));
                    ExecutionContext exe_ctx (GetExecutionContextRef());
                    m_addr_of_valobj_sp = ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(),
                                                                          ast, 
                                                                          ClangASTContext::CreatePointerType (ast, clang_type),
                                                                          ConstString (name.c_str()),
                                                                          addr, 
                                                                          eAddressTypeInvalid,
                                                                          m_data.GetAddressByteSize());
                }
            }
            break;
        }
    }
    else
    {
        StreamString expr_path_strm;
        GetExpressionPath(expr_path_strm, true);
        error.SetErrorStringWithFormat("'%s' doesn't have a valid address", expr_path_strm.GetString().c_str());
    }
    
    return m_addr_of_valobj_sp;
}

ValueObjectSP
ValueObject::Cast (const ClangASTType &clang_ast_type)
{
    return ValueObjectCast::Create (*this, GetName(), clang_ast_type);
}

ValueObjectSP
ValueObject::CastPointerType (const char *name, ClangASTType &clang_ast_type)
{
    ValueObjectSP valobj_sp;
    AddressType address_type;
    addr_t ptr_value = GetPointerValue (&address_type);
    
    if (ptr_value != LLDB_INVALID_ADDRESS)
    {
        Address ptr_addr (ptr_value);
        ExecutionContext exe_ctx (GetExecutionContextRef());
        valobj_sp = ValueObjectMemory::Create (exe_ctx.GetBestExecutionContextScope(),
                                               name, 
                                               ptr_addr, 
                                               clang_ast_type);
    }
    return valobj_sp;    
}

ValueObjectSP
ValueObject::CastPointerType (const char *name, TypeSP &type_sp)
{
    ValueObjectSP valobj_sp;
    AddressType address_type;
    addr_t ptr_value = GetPointerValue (&address_type);
    
    if (ptr_value != LLDB_INVALID_ADDRESS)
    {
        Address ptr_addr (ptr_value);
        ExecutionContext exe_ctx (GetExecutionContextRef());
        valobj_sp = ValueObjectMemory::Create (exe_ctx.GetBestExecutionContextScope(),
                                               name, 
                                               ptr_addr, 
                                               type_sp);
    }
    return valobj_sp;
}

ValueObject::EvaluationPoint::EvaluationPoint () :
    m_mod_id(),
    m_exe_ctx_ref(),
    m_needs_update (true),
    m_first_update (true)
{
}

ValueObject::EvaluationPoint::EvaluationPoint (ExecutionContextScope *exe_scope, bool use_selected):
    m_mod_id(),
    m_exe_ctx_ref(),
    m_needs_update (true),
    m_first_update (true)
{
    ExecutionContext exe_ctx(exe_scope);
    TargetSP target_sp (exe_ctx.GetTargetSP());
    if (target_sp)
    {
        m_exe_ctx_ref.SetTargetSP (target_sp);
        ProcessSP process_sp (exe_ctx.GetProcessSP());
        if (!process_sp)
            process_sp = target_sp->GetProcessSP();
        
        if (process_sp)
        {
            m_mod_id = process_sp->GetModID();
            m_exe_ctx_ref.SetProcessSP (process_sp);
            
            ThreadSP thread_sp (exe_ctx.GetThreadSP());
            
            if (!thread_sp)
            {
                if (use_selected)
                    thread_sp = process_sp->GetThreadList().GetSelectedThread();
            }
                
            if (thread_sp)
            {
                m_exe_ctx_ref.SetThreadSP(thread_sp);
                
                StackFrameSP frame_sp (exe_ctx.GetFrameSP());
                if (!frame_sp)
                {
                    if (use_selected)
                        frame_sp = thread_sp->GetSelectedFrame();
                }
                if (frame_sp)
                    m_exe_ctx_ref.SetFrameSP(frame_sp);
            }
        }
    }
}

ValueObject::EvaluationPoint::EvaluationPoint (const ValueObject::EvaluationPoint &rhs) :
    m_mod_id(),
    m_exe_ctx_ref(rhs.m_exe_ctx_ref),
    m_needs_update (true),
    m_first_update (true)
{
}

ValueObject::EvaluationPoint::~EvaluationPoint () 
{
}

// This function checks the EvaluationPoint against the current process state.  If the current
// state matches the evaluation point, or the evaluation point is already invalid, then we return
// false, meaning "no change".  If the current state is different, we update our state, and return
// true meaning "yes, change".  If we did see a change, we also set m_needs_update to true, so 
// future calls to NeedsUpdate will return true.
// exe_scope will be set to the current execution context scope.

bool
ValueObject::EvaluationPoint::SyncWithProcessState()
{

    // Start with the target, if it is NULL, then we're obviously not going to get any further:
    ExecutionContext exe_ctx(m_exe_ctx_ref.Lock());
    
    if (exe_ctx.GetTargetPtr() == NULL)
        return false;
    
    // If we don't have a process nothing can change.
    Process *process = exe_ctx.GetProcessPtr();
    if (process == NULL)
        return false;
        
    // If our stop id is the current stop ID, nothing has changed:
    ProcessModID current_mod_id = process->GetModID();
    
    // If the current stop id is 0, either we haven't run yet, or the process state has been cleared.
    // In either case, we aren't going to be able to sync with the process state.
    if (current_mod_id.GetStopID() == 0)
        return false;
    
    bool changed = false;
    const bool was_valid = m_mod_id.IsValid();
    if (was_valid)
    {
        if (m_mod_id == current_mod_id)
        {
            // Everything is already up to date in this object, no need to 
            // update the execution context scope.
            changed = false;
        }
        else
        {
            m_mod_id = current_mod_id;
            m_needs_update = true;
            changed = true;
        }       
    }
    
    // Now re-look up the thread and frame in case the underlying objects have gone away & been recreated.
    // That way we'll be sure to return a valid exe_scope.
    // If we used to have a thread or a frame but can't find it anymore, then mark ourselves as invalid.
    
    if (m_exe_ctx_ref.HasThreadRef())
    {
        ThreadSP thread_sp (m_exe_ctx_ref.GetThreadSP());
        if (thread_sp)
        {
            if (m_exe_ctx_ref.HasFrameRef())
            {
                StackFrameSP frame_sp (m_exe_ctx_ref.GetFrameSP());
                if (!frame_sp)
                {
                    // We used to have a frame, but now it is gone
                    SetInvalid();
                    changed = was_valid;
                }
            }
        }
        else
        {
            // We used to have a thread, but now it is gone
            SetInvalid();
            changed = was_valid;
        }

    }
    return changed;
}

void
ValueObject::EvaluationPoint::SetUpdated ()
{
    ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
    if (process_sp)
        m_mod_id = process_sp->GetModID();
    m_first_update = false;
    m_needs_update = false;
}
        

//bool
//ValueObject::EvaluationPoint::SetContext (ExecutionContextScope *exe_scope)
//{
//    if (!IsValid())
//        return false;
//    
//    bool needs_update = false;
//    
//    // The target has to be non-null, and the 
//    Target *target = exe_scope->CalculateTarget();
//    if (target != NULL)
//    {
//        Target *old_target = m_target_sp.get();
//        assert (target == old_target);
//        Process *process = exe_scope->CalculateProcess();
//        if (process != NULL)
//        {
//            // FOR NOW - assume you can't update variable objects across process boundaries.
//            Process *old_process = m_process_sp.get();
//            assert (process == old_process);
//            ProcessModID current_mod_id = process->GetModID();
//            if (m_mod_id != current_mod_id)
//            {
//                needs_update = true;
//                m_mod_id = current_mod_id;
//            }
//            // See if we're switching the thread or stack context.  If no thread is given, this is
//            // being evaluated in a global context.            
//            Thread *thread = exe_scope->CalculateThread();
//            if (thread != NULL)
//            {
//                user_id_t new_thread_index = thread->GetIndexID();
//                if (new_thread_index != m_thread_id)
//                {
//                    needs_update = true;
//                    m_thread_id = new_thread_index;
//                    m_stack_id.Clear();
//                }
//                
//                StackFrame *new_frame = exe_scope->CalculateStackFrame();
//                if (new_frame != NULL)
//                {
//                    if (new_frame->GetStackID() != m_stack_id)
//                    {
//                        needs_update = true;
//                        m_stack_id = new_frame->GetStackID();
//                    }
//                }
//                else
//                {
//                    m_stack_id.Clear();
//                    needs_update = true;
//                }
//            }
//            else
//            {
//                // If this had been given a thread, and now there is none, we should update.
//                // Otherwise we don't have to do anything.
//                if (m_thread_id != LLDB_INVALID_UID)
//                {
//                    m_thread_id = LLDB_INVALID_UID;
//                    m_stack_id.Clear();
//                    needs_update = true;
//                }
//            }
//        }
//        else
//        {
//            // If there is no process, then we don't need to update anything.
//            // But if we're switching from having a process to not, we should try to update.
//            if (m_process_sp.get() != NULL)
//            {
//                needs_update = true;
//                m_process_sp.reset();
//                m_thread_id = LLDB_INVALID_UID;
//                m_stack_id.Clear();
//            }
//        }
//    }
//    else
//    {
//        // If there's no target, nothing can change so we don't need to update anything.
//        // But if we're switching from having a target to not, we should try to update.
//        if (m_target_sp.get() != NULL)
//        {
//            needs_update = true;
//            m_target_sp.reset();
//            m_process_sp.reset();
//            m_thread_id = LLDB_INVALID_UID;
//            m_stack_id.Clear();
//        }
//    }
//    if (!m_needs_update)
//        m_needs_update = needs_update;
//        
//    return needs_update;
//}

void
ValueObject::ClearUserVisibleData(uint32_t clear_mask)
{
    if ((clear_mask & eClearUserVisibleDataItemsValue) == eClearUserVisibleDataItemsValue)
        m_value_str.clear();
    
    if ((clear_mask & eClearUserVisibleDataItemsLocation) == eClearUserVisibleDataItemsLocation)
        m_location_str.clear();
    
    if ((clear_mask & eClearUserVisibleDataItemsSummary) == eClearUserVisibleDataItemsSummary)
    {
        m_summary_str.clear();
    }
    
    if ((clear_mask & eClearUserVisibleDataItemsDescription) == eClearUserVisibleDataItemsDescription)
        m_object_desc_str.clear();
    
    if ((clear_mask & eClearUserVisibleDataItemsSyntheticChildren) == eClearUserVisibleDataItemsSyntheticChildren)
    {
            if (m_synthetic_value)
                m_synthetic_value = NULL;
    }
}

SymbolContextScope *
ValueObject::GetSymbolContextScope()
{
    if (m_parent)
    {
        if (!m_parent->IsPointerOrReferenceType())
            return m_parent->GetSymbolContextScope();
    }
    return NULL;
}

lldb::ValueObjectSP
ValueObject::CreateValueObjectFromExpression (const char* name,
                                              const char* expression,
                                              const ExecutionContext& exe_ctx)
{
    lldb::ValueObjectSP retval_sp;
    lldb::TargetSP target_sp(exe_ctx.GetTargetSP());
    if (!target_sp)
        return retval_sp;
    if (!expression || !*expression)
        return retval_sp;
    target_sp->EvaluateExpression (expression,
                                   exe_ctx.GetFrameSP().get(),
                                   retval_sp);
    if (retval_sp && name && *name)
        retval_sp->SetName(ConstString(name));
    return retval_sp;
}

lldb::ValueObjectSP
ValueObject::CreateValueObjectFromAddress (const char* name,
                                           uint64_t address,
                                           const ExecutionContext& exe_ctx,
                                           ClangASTType type)
{
    ClangASTType pointer_type(type.GetASTContext(),type.GetPointerType());
    lldb::DataBufferSP buffer(new lldb_private::DataBufferHeap(&address,sizeof(lldb::addr_t)));
    lldb::ValueObjectSP ptr_result_valobj_sp(ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(),
                                                                             pointer_type.GetASTContext(),
                                                                             pointer_type.GetOpaqueQualType(),
                                                                             ConstString(name),
                                                                             buffer,
                                                                             lldb::endian::InlHostByteOrder(),
                                                                             exe_ctx.GetAddressByteSize()));
    if (ptr_result_valobj_sp)
    {
        ptr_result_valobj_sp->GetValue().SetValueType(Value::eValueTypeLoadAddress);
        Error err;
        ptr_result_valobj_sp = ptr_result_valobj_sp->Dereference(err);
        if (ptr_result_valobj_sp && name && *name)
            ptr_result_valobj_sp->SetName(ConstString(name));
    }
    return ptr_result_valobj_sp;
}

lldb::ValueObjectSP
ValueObject::CreateValueObjectFromData (const char* name,
                                        DataExtractor& data,
                                        const ExecutionContext& exe_ctx,
                                        ClangASTType type)
{
    lldb::ValueObjectSP new_value_sp;
    new_value_sp = ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(),
                                                   type.GetASTContext() ,
                                                   type.GetOpaqueQualType(),
                                                   ConstString(name),
                                                   data,
                                                   LLDB_INVALID_ADDRESS);
    new_value_sp->SetAddressTypeOfChildren(eAddressTypeLoad);
    if (new_value_sp && name && *name)
        new_value_sp->SetName(ConstString(name));
    return new_value_sp;
}

ModuleSP
ValueObject::GetModule ()
{
    ValueObject* root(GetRoot());
    if (root != this)
        return root->GetModule();
    return lldb::ModuleSP();
}

ValueObject*
ValueObject::GetRoot ()
{
    if (m_root)
        return m_root;
    ValueObject* parent = m_parent;
    if (!parent)
        return (m_root = this);
    while (parent->m_parent)
    {
        if (parent->m_root)
            return (m_root = parent->m_root);
        parent = parent->m_parent;
    }
    return (m_root = parent);
}

AddressType
ValueObject::GetAddressTypeOfChildren()
{
    if (m_address_type_of_ptr_or_ref_children == eAddressTypeInvalid)
    {
        ValueObject* root(GetRoot());
        if (root != this)
            return root->GetAddressTypeOfChildren();
    }
    return m_address_type_of_ptr_or_ref_children;
}

lldb::DynamicValueType
ValueObject::GetDynamicValueType ()
{
    ValueObject* with_dv_info = this;
    while (with_dv_info)
    {
        if (with_dv_info->HasDynamicValueTypeInfo())
            return with_dv_info->GetDynamicValueTypeImpl();
        with_dv_info = with_dv_info->m_parent;
    }
    return lldb::eNoDynamicValues;
}
lldb::Format
ValueObject::GetFormat () const
{
    const ValueObject* with_fmt_info = this;
    while (with_fmt_info)
    {
        if (with_fmt_info->m_format != lldb::eFormatDefault)
            return with_fmt_info->m_format;
        with_fmt_info = with_fmt_info->m_parent;
    }
    return m_format;
}
