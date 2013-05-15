//===-- ValueObject.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObject_h_
#define liblldb_ValueObject_h_

// C Includes
// C++ Includes
#include <initializer_list>
#include <map>
#include <vector>
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/Value.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackID.h"
#include "lldb/Utility/SharedCluster.h"

namespace lldb_private {

/// ValueObject:
///
/// This abstract class provides an interface to a particular value, be it a register, a local or global variable,
/// that is evaluated in some particular scope.  The ValueObject also has the capibility of being the "child" of
/// some other variable object, and in turn of having children.  
/// If a ValueObject is a root variable object - having no parent - then it must be constructed with respect to some
/// particular ExecutionContextScope.  If it is a child, it inherits the ExecutionContextScope from its parent.
/// The ValueObject will update itself if necessary before fetching its value, summary, object description, etc.
/// But it will always update itself in the ExecutionContextScope with which it was originally created.

/// A brief note on life cycle management for ValueObjects.  This is a little tricky because a ValueObject can contain
/// various other ValueObjects - the Dynamic Value, its children, the dereference value, etc.  Any one of these can be
/// handed out as a shared pointer, but for that contained value object to be valid, the root object and potentially other
/// of the value objects need to stay around.  
/// We solve this problem by handing out shared pointers to the Value Object and any of its dependents using a shared
/// ClusterManager.  This treats each shared pointer handed out for the entire cluster as a reference to the whole
/// cluster.  The whole cluster will stay around until the last reference is released.
///
/// The ValueObject mostly handle this automatically, if a value object is made with a Parent ValueObject, then it adds
/// itself to the ClusterManager of the parent.

/// It does mean that external to the ValueObjects we should only ever make available ValueObjectSP's, never ValueObjects 
/// or pointers to them.  So all the "Root level" ValueObject derived constructors should be private, and 
/// should implement a Create function that new's up object and returns a Shared Pointer that it gets from the GetSP() method.
///
/// However, if you are making an derived ValueObject that will be contained in a parent value object, you should just
/// hold onto a pointer to it internally, and by virtue of passing the parent ValueObject into its constructor, it will
/// be added to the ClusterManager for the parent.  Then if you ever hand out a Shared Pointer to the contained ValueObject,
/// just do so by calling GetSP() on the contained object.

class ValueObject : public UserID
{
public:
    
    enum GetExpressionPathFormat
    {
        eGetExpressionPathFormatDereferencePointers = 1,
        eGetExpressionPathFormatHonorPointers
    };
    
    enum ValueObjectRepresentationStyle
    {
        eValueObjectRepresentationStyleValue = 1,
        eValueObjectRepresentationStyleSummary,
        eValueObjectRepresentationStyleLanguageSpecific,
        eValueObjectRepresentationStyleLocation,
        eValueObjectRepresentationStyleChildrenCount,
        eValueObjectRepresentationStyleType
    };
    
    enum ExpressionPathScanEndReason
    {
        eExpressionPathScanEndReasonEndOfString = 1,           // out of data to parse
        eExpressionPathScanEndReasonNoSuchChild,               // child element not found
        eExpressionPathScanEndReasonEmptyRangeNotAllowed,      // [] only allowed for arrays
        eExpressionPathScanEndReasonDotInsteadOfArrow,         // . used when -> should be used
        eExpressionPathScanEndReasonArrowInsteadOfDot,         // -> used when . should be used
        eExpressionPathScanEndReasonFragileIVarNotAllowed,     // ObjC ivar expansion not allowed
        eExpressionPathScanEndReasonRangeOperatorNotAllowed,   // [] not allowed by options
        eExpressionPathScanEndReasonRangeOperatorInvalid,      // [] not valid on objects other than scalars, pointers or arrays
        eExpressionPathScanEndReasonArrayRangeOperatorMet,     // [] is good for arrays, but I cannot parse it
        eExpressionPathScanEndReasonBitfieldRangeOperatorMet,  // [] is good for bitfields, but I cannot parse after it
        eExpressionPathScanEndReasonUnexpectedSymbol,          // something is malformed in the expression
        eExpressionPathScanEndReasonTakingAddressFailed,       // impossible to apply & operator
        eExpressionPathScanEndReasonDereferencingFailed,       // impossible to apply * operator
        eExpressionPathScanEndReasonRangeOperatorExpanded,     // [] was expanded into a VOList
        eExpressionPathScanEndReasonSyntheticValueMissing,     // getting the synthetic children failed
        eExpressionPathScanEndReasonUnknown = 0xFFFF
    };
    
    enum ExpressionPathEndResultType
    {
        eExpressionPathEndResultTypePlain = 1,                 // anything but...
        eExpressionPathEndResultTypeBitfield,                  // a bitfield
        eExpressionPathEndResultTypeBoundedRange,              // a range [low-high]
        eExpressionPathEndResultTypeUnboundedRange,            // a range []
        eExpressionPathEndResultTypeValueObjectList,           // several items in a VOList
        eExpressionPathEndResultTypeInvalid = 0xFFFF
    };
    
    enum ExpressionPathAftermath
    {
        eExpressionPathAftermathNothing = 1,               // just return it
        eExpressionPathAftermathDereference,               // dereference the target
        eExpressionPathAftermathTakeAddress                // take target's address
    };
    
    enum ClearUserVisibleDataItems
    {
        eClearUserVisibleDataItemsNothing = 1u << 0,
        eClearUserVisibleDataItemsValue = 1u << 1,
        eClearUserVisibleDataItemsSummary = 1u << 2,
        eClearUserVisibleDataItemsLocation = 1u << 3,
        eClearUserVisibleDataItemsDescription = 1u << 4,
        eClearUserVisibleDataItemsSyntheticChildren = 1u << 5,
        eClearUserVisibleDataItemsAllStrings = eClearUserVisibleDataItemsValue | eClearUserVisibleDataItemsSummary | eClearUserVisibleDataItemsLocation | eClearUserVisibleDataItemsDescription,
        eClearUserVisibleDataItemsAll = 0xFFFF
    };
    
    struct GetValueForExpressionPathOptions
    {
        bool m_check_dot_vs_arrow_syntax;
        bool m_no_fragile_ivar;
        bool m_allow_bitfields_syntax;
        bool m_no_synthetic_children;
        
        GetValueForExpressionPathOptions(bool dot = false,
                                         bool no_ivar = false,
                                         bool bitfield = true,
                                         bool no_synth = false) :
            m_check_dot_vs_arrow_syntax(dot),
            m_no_fragile_ivar(no_ivar),
            m_allow_bitfields_syntax(bitfield),
            m_no_synthetic_children(no_synth)
        {
        }
        
        GetValueForExpressionPathOptions&
        DoCheckDotVsArrowSyntax()
        {
            m_check_dot_vs_arrow_syntax = true;
            return *this;
        }
        
        GetValueForExpressionPathOptions&
        DontCheckDotVsArrowSyntax()
        {
            m_check_dot_vs_arrow_syntax = false;
            return *this;
        }
        
        GetValueForExpressionPathOptions&
        DoAllowFragileIVar()
        {
            m_no_fragile_ivar = false;
            return *this;
        }
        
        GetValueForExpressionPathOptions&
        DontAllowFragileIVar()
        {
            m_no_fragile_ivar = true;
            return *this;
        }

        GetValueForExpressionPathOptions&
        DoAllowBitfieldSyntax()
        {
            m_allow_bitfields_syntax = true;
            return *this;
        }
        
        GetValueForExpressionPathOptions&
        DontAllowBitfieldSyntax()
        {
            m_allow_bitfields_syntax = false;
            return *this;
        }
        
        GetValueForExpressionPathOptions&
        DoAllowSyntheticChildren()
        {
            m_no_synthetic_children = false;
            return *this;
        }
        
        GetValueForExpressionPathOptions&
        DontAllowSyntheticChildren()
        {
            m_no_synthetic_children = true;
            return *this;
        }
        
        static const GetValueForExpressionPathOptions
        DefaultOptions()
        {
            static GetValueForExpressionPathOptions g_default_options;
            
            return g_default_options;
        }

    };
    
    struct DumpValueObjectOptions
    {
        uint32_t m_max_ptr_depth;
        uint32_t m_max_depth;
        bool m_show_types;
        bool m_show_location;
        bool m_use_objc;
        lldb::DynamicValueType m_use_dynamic;
        bool m_use_synthetic;
        bool m_scope_already_checked;
        bool m_flat_output;
        uint32_t m_omit_summary_depth;
        bool m_ignore_cap;
        lldb::Format m_format;
        lldb::TypeSummaryImplSP m_summary_sp;
        std::string m_root_valobj_name;
        bool m_hide_root_type;
        bool m_hide_name;
        bool m_hide_value;
        
        DumpValueObjectOptions() :
            m_max_ptr_depth(0),
            m_max_depth(UINT32_MAX),
            m_show_types(false),
            m_show_location(false),
            m_use_objc(false),
            m_use_dynamic(lldb::eNoDynamicValues),
            m_use_synthetic(true),
            m_scope_already_checked(false),
            m_flat_output(false),
            m_omit_summary_depth(0),
            m_ignore_cap(false), 
            m_format (lldb::eFormatDefault),
            m_summary_sp(),
            m_root_valobj_name(),
            m_hide_root_type(false),  // provide a special compact display for "po"
            m_hide_name(false), // provide a special compact display for "po"
            m_hide_value(false) // provide a special compact display for "po"
        {}
        
        static const DumpValueObjectOptions
        DefaultOptions()
        {
            static DumpValueObjectOptions g_default_options;
            
            return g_default_options;
        }
        
        DumpValueObjectOptions (const DumpValueObjectOptions& rhs) :
            m_max_ptr_depth(rhs.m_max_ptr_depth),
            m_max_depth(rhs.m_max_depth),
            m_show_types(rhs.m_show_types),
            m_show_location(rhs.m_show_location),
            m_use_objc(rhs.m_use_objc),
            m_use_dynamic(rhs.m_use_dynamic),
            m_use_synthetic(rhs.m_use_synthetic),
            m_scope_already_checked(rhs.m_scope_already_checked),
            m_flat_output(rhs.m_flat_output),
            m_omit_summary_depth(rhs.m_omit_summary_depth),
            m_ignore_cap(rhs.m_ignore_cap),
            m_format(rhs.m_format),
            m_summary_sp(rhs.m_summary_sp),
            m_root_valobj_name(rhs.m_root_valobj_name),
            m_hide_root_type(rhs.m_hide_root_type),
            m_hide_name(rhs.m_hide_name),
            m_hide_value(rhs.m_hide_value)
        {}
        
        DumpValueObjectOptions&
        SetMaximumPointerDepth(uint32_t depth = 0)
        {
            m_max_ptr_depth = depth;
            return *this;
        }
        
        DumpValueObjectOptions&
        SetMaximumDepth(uint32_t depth = 0)
        {
            m_max_depth = depth;
            return *this;
        }
        
        DumpValueObjectOptions&
        SetShowTypes(bool show = false)
        {
            m_show_types = show;
            return *this;
        }
        
        DumpValueObjectOptions&
        SetShowLocation(bool show = false)
        {
            m_show_location = show;
            return *this;
        }

        DumpValueObjectOptions&
        SetUseObjectiveC(bool use = false)
        {
            m_use_objc = use;
            return *this;
        }
    
        DumpValueObjectOptions&
        SetShowSummary(bool show = true)
        {
            if (show == false)
                SetOmitSummaryDepth(UINT32_MAX);
            else
                SetOmitSummaryDepth(0);
            return *this;
        }
        
        DumpValueObjectOptions&
        SetUseDynamicType(lldb::DynamicValueType dyn = lldb::eNoDynamicValues)
        {
            m_use_dynamic = dyn;
            return *this;
        }
        
        DumpValueObjectOptions&
        SetUseSyntheticValue(bool use_synthetic = true)
        {
            m_use_synthetic = use_synthetic;
            return *this;
        }

        DumpValueObjectOptions&
        SetScopeChecked(bool check = true)
        {
            m_scope_already_checked = check;
            return *this;
        }
        
        DumpValueObjectOptions&
        SetFlatOutput(bool flat = false)
        {
            m_flat_output = flat;
            return *this;
        }
        
        DumpValueObjectOptions&
        SetOmitSummaryDepth(uint32_t depth = 0)
        {
            m_omit_summary_depth = depth;
            return *this;
        }
        
        DumpValueObjectOptions&
        SetIgnoreCap(bool ignore = false)
        {
            m_ignore_cap = ignore;
            return *this;
        }

        DumpValueObjectOptions&
        SetRawDisplay(bool raw = false)
        {
            if (raw)
            {
                SetUseSyntheticValue(false);
                SetOmitSummaryDepth(UINT32_MAX);
                SetIgnoreCap(true);
                SetHideName(false);
                SetHideValue(false);
            }
            else
            {
                SetUseSyntheticValue(true);
                SetOmitSummaryDepth(0);
                SetIgnoreCap(false);
                SetHideName(false);
                SetHideValue(false);
            }
            return *this;
        }

        DumpValueObjectOptions&
        SetFormat (lldb::Format format = lldb::eFormatDefault)
        {
            m_format = format;
            return *this;
        }
        
        DumpValueObjectOptions&
        SetSummary (lldb::TypeSummaryImplSP summary = lldb::TypeSummaryImplSP())
        {
            m_summary_sp = summary;
            return *this;
        }
        
        DumpValueObjectOptions&
        SetRootValueObjectName (const char* name = NULL)
        {
            if (name)
                m_root_valobj_name.assign(name);
            else
                m_root_valobj_name.clear();
            return *this;
        }
                
        DumpValueObjectOptions&
        SetHideRootType (bool hide_root_type = false)
        {
            m_hide_root_type = hide_root_type;
            return *this;
        }
        
        DumpValueObjectOptions&
        SetHideName (bool hide_name = false)
        {
            m_hide_name = hide_name;
            return *this;
        }

        DumpValueObjectOptions&
        SetHideValue (bool hide_value = false)
        {
            m_hide_value = hide_value;
            return *this;
        }        
    };

    class EvaluationPoint
    {
    public:
        
        EvaluationPoint ();
        
        EvaluationPoint (ExecutionContextScope *exe_scope, bool use_selected = false);
        
        EvaluationPoint (const EvaluationPoint &rhs);
        
        ~EvaluationPoint ();
        
        const ExecutionContextRef &
        GetExecutionContextRef() const
        {
            return m_exe_ctx_ref;
        }

        // Set the EvaluationPoint to the values in exe_scope,
        // Return true if the Evaluation Point changed.
        // Since the ExecutionContextScope is always going to be valid currently, 
        // the Updated Context will also always be valid.
        
//        bool
//        SetContext (ExecutionContextScope *exe_scope);
        
        void
        SetIsConstant ()
        {
            SetUpdated();
            m_mod_id.SetInvalid();
        }
        
        bool
        IsConstant () const
        {
            return !m_mod_id.IsValid();
        }
        
        ProcessModID
        GetModID () const
        {
            return m_mod_id;
        }

        void
        SetUpdateID (ProcessModID new_id)
        {
            m_mod_id = new_id;
        }
        
        bool
        IsFirstEvaluation () const
        {
            return m_first_update;
        }
        
        void
        SetNeedsUpdate ()
        {
            m_needs_update = true;
        }
        
        void
        SetUpdated ();
        
        bool
        NeedsUpdating()
        {
            SyncWithProcessState();
            return m_needs_update;
        }
        
        bool
        IsValid ()
        {
            if (!m_mod_id.IsValid())
                return false;
            else if (SyncWithProcessState ())
            {
                if (!m_mod_id.IsValid())
                    return false;
            }
            return true;
        }
        
        void
        SetInvalid ()
        {
            // Use the stop id to mark us as invalid, leave the thread id and the stack id around for logging and
            // history purposes.
            m_mod_id.SetInvalid();
            
            // Can't update an invalid state.
            m_needs_update = false;
            
        }
        
    private:
        bool
        SyncWithProcessState ();
                
        ProcessModID m_mod_id; // This is the stop id when this ValueObject was last evaluated.
        ExecutionContextRef m_exe_ctx_ref;
        bool m_needs_update;
        bool m_first_update;
    };

    const EvaluationPoint &
    GetUpdatePoint () const
    {
        return m_update_point;
    }
    
    EvaluationPoint &
    GetUpdatePoint ()
    {
        return m_update_point;
    }
    
    const ExecutionContextRef &
    GetExecutionContextRef() const
    {
        return m_update_point.GetExecutionContextRef();
    }

    lldb::TargetSP
    GetTargetSP() const
    {
        return m_update_point.GetExecutionContextRef().GetTargetSP();
    }

    lldb::ProcessSP
    GetProcessSP() const
    {
        return m_update_point.GetExecutionContextRef().GetProcessSP();
    }

    lldb::ThreadSP
    GetThreadSP() const
    {
        return m_update_point.GetExecutionContextRef().GetThreadSP();
    }

    lldb::StackFrameSP
    GetFrameSP() const
    {
        return m_update_point.GetExecutionContextRef().GetFrameSP();
    }

    void
    SetNeedsUpdate ();
    
    virtual ~ValueObject();
    
    clang::ASTContext *
    GetClangAST ();
    
    lldb::clang_type_t
    GetClangType ();

    //------------------------------------------------------------------
    // Sublasses must implement the functions below.
    //------------------------------------------------------------------
    virtual uint64_t
    GetByteSize() = 0;

    virtual lldb::ValueType
    GetValueType() const = 0;

    //------------------------------------------------------------------
    // Sublasses can implement the functions below.
    //------------------------------------------------------------------
    virtual ConstString
    GetTypeName();
    
    virtual ConstString
    GetQualifiedTypeName();

    virtual lldb::LanguageType
    GetObjectRuntimeLanguage();

    virtual uint32_t
    GetTypeInfo (lldb::clang_type_t *pointee_or_element_clang_type = NULL);

    virtual bool
    IsPointerType ();
    
    virtual bool
    IsArrayType ();
    
    virtual bool
    IsScalarType ();

    virtual bool
    IsPointerOrReferenceType ();
    
    virtual bool
    IsPossibleDynamicType ();

    virtual bool
    IsObjCNil ();
    
    virtual bool
    IsBaseClass ()
    {
        return false;
    }
    
    virtual bool
    IsDereferenceOfParent ()
    {
        return false;
    }
    
    bool
    IsIntegerType (bool &is_signed);
    
    virtual bool
    GetBaseClassPath (Stream &s);

    virtual void
    GetExpressionPath (Stream &s, bool qualify_cxx_base_classes, GetExpressionPathFormat = eGetExpressionPathFormatDereferencePointers);
    
    lldb::ValueObjectSP
    GetValueForExpressionPath(const char* expression,
                              const char** first_unparsed = NULL,
                              ExpressionPathScanEndReason* reason_to_stop = NULL,
                              ExpressionPathEndResultType* final_value_type = NULL,
                              const GetValueForExpressionPathOptions& options = GetValueForExpressionPathOptions::DefaultOptions(),
                              ExpressionPathAftermath* final_task_on_target = NULL);
    
    int
    GetValuesForExpressionPath(const char* expression,
                               lldb::ValueObjectListSP& list,
                               const char** first_unparsed = NULL,
                               ExpressionPathScanEndReason* reason_to_stop = NULL,
                               ExpressionPathEndResultType* final_value_type = NULL,
                               const GetValueForExpressionPathOptions& options = GetValueForExpressionPathOptions::DefaultOptions(),
                               ExpressionPathAftermath* final_task_on_target = NULL);
    
    virtual bool
    IsInScope ()
    {
        return true;
    }

    virtual off_t
    GetByteOffset()
    {
        return 0;
    }

    virtual uint32_t
    GetBitfieldBitSize ()
    {
        return 0;
    }

    virtual uint32_t
    GetBitfieldBitOffset ()
    {
        return 0;
    }
    
    bool
    IsBitfield ()
    {
        return (GetBitfieldBitSize() != 0) || (GetBitfieldBitOffset() != 0);
    }
    
    virtual bool
    IsArrayItemForPointer()
    {
        return m_is_array_item_for_pointer;
    }
    
    virtual bool
    SetClangAST (clang::ASTContext *ast)
    {
        return false;
    }

    virtual const char *
    GetValueAsCString ();
    
    virtual bool
    GetValueAsCString (lldb::Format format,
                       std::string& destination);
    
    virtual uint64_t
    GetValueAsUnsigned (uint64_t fail_value, bool *success = NULL);

    virtual bool
    SetValueFromCString (const char *value_str, Error& error);
    
    // Return the module associated with this value object in case the
    // value is from an executable file and might have its data in
    // sections of the file. This can be used for variables.
    virtual lldb::ModuleSP
    GetModule();
    
    virtual ValueObject*
    GetRoot ();
    
    virtual bool
    GetDeclaration (Declaration &decl);

    //------------------------------------------------------------------
    // The functions below should NOT be modified by sublasses
    //------------------------------------------------------------------
    const Error &
    GetError();

    const ConstString &
    GetName() const;

    virtual lldb::ValueObjectSP
    GetChildAtIndex (size_t idx, bool can_create);

    // this will always create the children if necessary
    lldb::ValueObjectSP
    GetChildAtIndexPath (const std::initializer_list<size_t> &idxs,
                         size_t* index_of_error = NULL);
    
    lldb::ValueObjectSP
    GetChildAtIndexPath (const std::vector<size_t> &idxs,
                         size_t* index_of_error = NULL);
    
    lldb::ValueObjectSP
    GetChildAtIndexPath (const std::initializer_list< std::pair<size_t, bool> > &idxs,
                         size_t* index_of_error = NULL);

    lldb::ValueObjectSP
    GetChildAtIndexPath (const std::vector< std::pair<size_t, bool> > &idxs,
                         size_t* index_of_error = NULL);
    
    virtual lldb::ValueObjectSP
    GetChildMemberWithName (const ConstString &name, bool can_create);

    virtual size_t
    GetIndexOfChildWithName (const ConstString &name);

    size_t
    GetNumChildren ();

    const Value &
    GetValue() const;

    Value &
    GetValue();

    virtual bool
    ResolveValue (Scalar &scalar);
    
    virtual const char *
    GetLocationAsCString ();

    const char *
    GetSummaryAsCString ();
    
    bool
    GetSummaryAsCString (TypeSummaryImpl* summary_ptr,
                         std::string& destination);
    
    const char *
    GetObjectDescription ();
    
    bool
    HasSpecialPrintableRepresentation (ValueObjectRepresentationStyle val_obj_display,
                                       lldb::Format custom_format);
    
    enum PrintableRepresentationSpecialCases
    {
        ePrintableRepresentationSpecialCasesDisable = 0,
        ePrintableRepresentationSpecialCasesAllow = 1,
        ePrintableRepresentationSpecialCasesOnly = 3
    };
    
    bool
    DumpPrintableRepresentation (Stream& s,
                                 ValueObjectRepresentationStyle val_obj_display = eValueObjectRepresentationStyleSummary,
                                 lldb::Format custom_format = lldb::eFormatInvalid,
                                 PrintableRepresentationSpecialCases special = ePrintableRepresentationSpecialCasesAllow);
    bool
    GetValueIsValid () const;

    bool
    GetValueDidChange ();

    bool
    UpdateValueIfNeeded (bool update_format = true);
    
    bool
    UpdateFormatsIfNeeded();

    lldb::ValueObjectSP
    GetSP ()
    {
        return m_manager->GetSharedPointer(this);
    }
    
    void
    SetName (const ConstString &name);
    
    virtual lldb::addr_t
    GetAddressOf (bool scalar_is_load_address = true,
                  AddressType *address_type = NULL);
    
    lldb::addr_t
    GetPointerValue (AddressType *address_type = NULL);
    
    lldb::ValueObjectSP
    GetSyntheticChild (const ConstString &key) const;
    
    lldb::ValueObjectSP
    GetSyntheticArrayMember (size_t index, bool can_create);

    lldb::ValueObjectSP
    GetSyntheticArrayMemberFromPointer (size_t index, bool can_create);
    
    lldb::ValueObjectSP
    GetSyntheticArrayMemberFromArray (size_t index, bool can_create);
    
    lldb::ValueObjectSP
    GetSyntheticBitFieldChild (uint32_t from, uint32_t to, bool can_create);

    lldb::ValueObjectSP
    GetSyntheticExpressionPathChild(const char* expression, bool can_create);
    
    virtual lldb::ValueObjectSP
    GetSyntheticChildAtOffset(uint32_t offset, const ClangASTType& type, bool can_create);
    
    virtual lldb::ValueObjectSP
    GetDynamicValue (lldb::DynamicValueType valueType);
    
    lldb::DynamicValueType
    GetDynamicValueType ();
    
    virtual lldb::ValueObjectSP
    GetStaticValue ();
    
    virtual lldb::ValueObjectSP
    GetNonSyntheticValue ();
    
    lldb::ValueObjectSP
    GetSyntheticValue (bool use_synthetic = true);
    
    virtual bool
    HasSyntheticValue();
    
    virtual bool
    IsSynthetic() { return false; }
    
    virtual lldb::ValueObjectSP
    CreateConstantValue (const ConstString &name);

    virtual lldb::ValueObjectSP
    Dereference (Error &error);
    
    virtual lldb::ValueObjectSP
    AddressOf (Error &error);
    
    virtual lldb::addr_t
    GetLiveAddress()
    {
        return LLDB_INVALID_ADDRESS;
    }
    
    virtual void
    SetLiveAddress(lldb::addr_t addr = LLDB_INVALID_ADDRESS,
                   AddressType address_type = eAddressTypeLoad)
    {
    }

    virtual lldb::ValueObjectSP
    Cast (const ClangASTType &clang_ast_type);
    
    virtual lldb::ValueObjectSP
    CastPointerType (const char *name,
                     ClangASTType &ast_type);

    virtual lldb::ValueObjectSP
    CastPointerType (const char *name,
                     lldb::TypeSP &type_sp);

    // The backing bits of this value object were updated, clear any
    // descriptive string, so we know we have to refetch them
    virtual void
    ValueUpdated ()
    {
        ClearUserVisibleData(eClearUserVisibleDataItemsValue |
                             eClearUserVisibleDataItemsSummary |
                             eClearUserVisibleDataItemsDescription);
    }

    virtual bool
    IsDynamic ()
    {
        return false;
    }
    
    virtual SymbolContextScope *
    GetSymbolContextScope();
    
    static void
    DumpValueObject (Stream &s,
                     ValueObject *valobj);    
    static void
    DumpValueObject (Stream &s,
                     ValueObject *valobj,
                     const DumpValueObjectOptions& options);

    static lldb::ValueObjectSP
    CreateValueObjectFromExpression (const char* name,
                                     const char* expression,
                                     const ExecutionContext& exe_ctx);
    
    static lldb::ValueObjectSP
    CreateValueObjectFromAddress (const char* name,
                                  uint64_t address,
                                  const ExecutionContext& exe_ctx,
                                  ClangASTType type);
    
    static lldb::ValueObjectSP
    CreateValueObjectFromData (const char* name,
                               DataExtractor& data,
                               const ExecutionContext& exe_ctx,
                               ClangASTType type);
    
    static void
    LogValueObject (Log *log,
                    ValueObject *valobj);

    static void
    LogValueObject (Log *log,
                    ValueObject *valobj,
                    const DumpValueObjectOptions& options);


    // returns true if this is a char* or a char[]
    // if it is a char* and check_pointer is true,
    // it also checks that the pointer is valid
    bool
    IsCStringContainer (bool check_pointer = false);
    
    size_t
    ReadPointedString (Stream& s,
                       Error& error,
                       uint32_t max_length = 0,
                       bool honor_array = true,
                       lldb::Format item_format = lldb::eFormatCharArray);
    
    virtual size_t
    GetPointeeData (DataExtractor& data,
                    uint32_t item_idx = 0,
					uint32_t item_count = 1);
    
    virtual uint64_t
    GetData (DataExtractor& data);
    
    virtual bool
    SetData (DataExtractor &data, Error &error);

    bool
    GetIsConstant () const
    {
        return m_update_point.IsConstant();
    }
    
    void
    SetIsConstant ()
    {
        m_update_point.SetIsConstant();
    }

    lldb::Format
    GetFormat () const;
    
    void
    SetFormat (lldb::Format format)
    {
        if (format != m_format)
            ClearUserVisibleData(eClearUserVisibleDataItemsValue);
        m_format = format;
    }
    
    lldb::TypeSummaryImplSP
    GetSummaryFormat()
    {
        UpdateFormatsIfNeeded();
        return m_type_summary_sp;
    }
    
    void
    SetSummaryFormat(lldb::TypeSummaryImplSP format)
    {
        m_type_summary_sp = format;
        ClearUserVisibleData(eClearUserVisibleDataItemsSummary);
    }
    
    void
    SetValueFormat(lldb::TypeFormatImplSP format)
    {
        m_type_format_sp = format;
        ClearUserVisibleData(eClearUserVisibleDataItemsValue);
    }
    
    lldb::TypeFormatImplSP
    GetValueFormat()
    {
        UpdateFormatsIfNeeded();
        return m_type_format_sp;
    }
    
    void
    SetSyntheticChildren(const lldb::SyntheticChildrenSP &synth_sp)
    {
        if (synth_sp.get() == m_synthetic_children_sp.get())
            return;
        ClearUserVisibleData(eClearUserVisibleDataItemsSyntheticChildren);
        m_synthetic_children_sp = synth_sp;
    }
    
    lldb::SyntheticChildrenSP
    GetSyntheticChildren()
    {
        UpdateFormatsIfNeeded();
        return m_synthetic_children_sp;
    }

    // Use GetParent for display purposes, but if you want to tell the parent to update itself
    // then use m_parent.  The ValueObjectDynamicValue's parent is not the correct parent for
    // displaying, they are really siblings, so for display it needs to route through to its grandparent.
    virtual ValueObject *
    GetParent()
    {
        return m_parent;
    }

    virtual const ValueObject *
    GetParent() const
    {
        return m_parent;
    }

    ValueObject *
    GetNonBaseClassParent();

    void
    SetAddressTypeOfChildren(AddressType at)
    {
        m_address_type_of_ptr_or_ref_children = at;
    }
    
    AddressType
    GetAddressTypeOfChildren();
    
    void
    SetHasCompleteType()
    {
        m_did_calculate_complete_objc_class_type = true;
    }
    
    //------------------------------------------------------------------
    /// Find out if a SBValue might have children.
    ///
    /// This call is much more efficient than CalculateNumChildren() as
    /// it doesn't need to complete the underlying type. This is designed
    /// to be used in a UI environment in order to detect if the
    /// disclosure triangle should be displayed or not.
    ///
    /// This function returns true for class, union, structure,
    /// pointers, references, arrays and more. Again, it does so without
    /// doing any expensive type completion.
    ///
    /// @return
    ///     Returns \b true if the SBValue might have children, or \b
    ///     false otherwise.
    //------------------------------------------------------------------
    virtual bool
    MightHaveChildren();

protected:
    typedef ClusterManager<ValueObject> ValueObjectManager;
    
    class ChildrenManager
    {
    public:
        ChildrenManager() :
            m_mutex(Mutex::eMutexTypeRecursive),
            m_children(),
            m_children_count(0)
        {}
        
        bool
        HasChildAtIndex (size_t idx)
        {
            Mutex::Locker locker(m_mutex);
            ChildrenIterator iter = m_children.find(idx);
            ChildrenIterator end = m_children.end();
            return (iter != end);
        }
        
        ValueObject*
        GetChildAtIndex (size_t idx)
        {
            Mutex::Locker locker(m_mutex);
            ChildrenIterator iter = m_children.find(idx);
            ChildrenIterator end = m_children.end();
            if (iter == end)
                return NULL;
            else
                return iter->second;
        }
        
        void
        SetChildAtIndex (size_t idx, ValueObject* valobj)
        {
            ChildrenPair pair(idx,valobj); // we do not need to be mutex-protected to make a pair
            Mutex::Locker locker(m_mutex);
            m_children.insert(pair);
        }
        
        void
        SetChildrenCount (size_t count)
        {
            m_children_count = count;
        }
        
        size_t
        GetChildrenCount ()
        {
            return m_children_count;
        }
        
        void
        Clear()
        {
            m_children_count = 0;
            Mutex::Locker locker(m_mutex);
            m_children.clear();
        }
        
    private:
        typedef std::map<size_t, ValueObject*> ChildrenMap;
        typedef ChildrenMap::iterator ChildrenIterator;
        typedef ChildrenMap::value_type ChildrenPair;
        Mutex m_mutex;
        ChildrenMap m_children;
        size_t m_children_count;
    };

    //------------------------------------------------------------------
    // Classes that inherit from ValueObject can see and modify these
    //------------------------------------------------------------------
    ValueObject  *      m_parent;       // The parent value object, or NULL if this has no parent
    ValueObject  *      m_root;         // The root of the hierarchy for this ValueObject (or NULL if never calculated)
    EvaluationPoint     m_update_point; // Stores both the stop id and the full context at which this value was last 
                                        // updated.  When we are asked to update the value object, we check whether
                                        // the context & stop id are the same before updating.
    ConstString         m_name;         // The name of this object
    DataExtractor       m_data;         // A data extractor that can be used to extract the value.
    Value               m_value;
    Error               m_error;        // An error object that can describe any errors that occur when updating values.
    std::string         m_value_str;    // Cached value string that will get cleared if/when the value is updated.
    std::string         m_old_value_str;// Cached old value string from the last time the value was gotten
    std::string         m_location_str; // Cached location string that will get cleared if/when the value is updated.
    std::string         m_summary_str;  // Cached summary string that will get cleared if/when the value is updated.
    std::string         m_object_desc_str; // Cached result of the "object printer".  This differs from the summary
                                              // in that the summary is consed up by us, the object_desc_string is builtin.

    ClangASTType        m_override_type;// If the type of the value object should be overridden, the type to impose.
    
    ValueObjectManager *m_manager;      // This object is managed by the root object (any ValueObject that gets created
                                        // without a parent.)  The manager gets passed through all the generations of
                                        // dependent objects, and will keep the whole cluster of objects alive as long
                                        // as a shared pointer to any of them has been handed out.  Shared pointers to
                                        // value objects must always be made with the GetSP method.

    ChildrenManager                      m_children;
    std::map<ConstString, ValueObject *> m_synthetic_children;
    
    ValueObject*                         m_dynamic_value;
    ValueObject*                         m_synthetic_value;
    ValueObject*                         m_deref_valobj;
    
    lldb::ValueObjectSP m_addr_of_valobj_sp; // We have to hold onto a shared pointer to this one because it is created
                                             // as an independent ValueObjectConstResult, which isn't managed by us.

    lldb::Format                m_format;
    uint32_t                    m_last_format_mgr_revision;
    lldb::TypeSummaryImplSP     m_type_summary_sp;
    lldb::TypeFormatImplSP      m_type_format_sp;
    lldb::SyntheticChildrenSP   m_synthetic_children_sp;
    ProcessModID                m_user_id_of_forced_summary;
    AddressType                 m_address_type_of_ptr_or_ref_children;
    
    bool                m_value_is_valid:1,
                        m_value_did_change:1,
                        m_children_count_valid:1,
                        m_old_value_valid:1,
                        m_is_deref_of_parent:1,
                        m_is_array_item_for_pointer:1,
                        m_is_bitfield_for_scalar:1,
                        m_is_child_at_offset:1,
                        m_is_getting_summary:1,
                        m_did_calculate_complete_objc_class_type:1;
    
    friend class ClangExpressionDeclMap;  // For GetValue
    friend class ClangExpressionVariable; // For SetName
    friend class Target;                  // For SetName
    friend class ValueObjectConstResultImpl;

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    
    // Use the no-argument constructor to make a constant variable object (with no ExecutionContextScope.)
    
    ValueObject();
    
    // Use this constructor to create a "root variable object".  The ValueObject will be locked to this context
    // through-out its lifespan.
    
    ValueObject (ExecutionContextScope *exe_scope,
                 AddressType child_ptr_or_ref_addr_type = eAddressTypeLoad);
    
    // Use this constructor to create a ValueObject owned by another ValueObject.  It will inherit the ExecutionContext
    // of its parent.
    
    ValueObject (ValueObject &parent);

    ValueObjectManager *
    GetManager()
    {
        return m_manager;
    }
    
    virtual bool
    UpdateValue () = 0;

    virtual void
    CalculateDynamicValue (lldb::DynamicValueType use_dynamic);
    
    virtual lldb::DynamicValueType
    GetDynamicValueTypeImpl ()
    {
        return lldb::eNoDynamicValues;
    }
    
    virtual bool
    HasDynamicValueTypeInfo ()
    {
        return false;
    }
    
    virtual void
    CalculateSyntheticValue (bool use_synthetic = true);
    
    // Should only be called by ValueObject::GetChildAtIndex()
    // Returns a ValueObject managed by this ValueObject's manager.
    virtual ValueObject *
    CreateChildAtIndex (size_t idx, bool synthetic_array_member, int32_t synthetic_index);

    // Should only be called by ValueObject::GetNumChildren()
    virtual size_t
    CalculateNumChildren() = 0;

    void
    SetNumChildren (size_t num_children);

    void
    SetValueDidChange (bool value_changed);

    void
    SetValueIsValid (bool valid);
    
    void
    ClearUserVisibleData(uint32_t items = ValueObject::eClearUserVisibleDataItemsAllStrings);
    
    void
    AddSyntheticChild (const ConstString &key,
                       ValueObject *valobj);
    
    DataExtractor &
    GetDataExtractor ();
    
    void
    ClearDynamicTypeInformation ();
    
    //------------------------------------------------------------------
    // Sublasses must implement the functions below.
    //------------------------------------------------------------------
    
    virtual clang::ASTContext *
    GetClangASTImpl () = 0;
    
    virtual lldb::clang_type_t
    GetClangTypeImpl () = 0;
    
    const char *
    GetLocationAsCStringImpl (const Value& value,
                              const DataExtractor& data);
    
private:
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    
    virtual ClangASTType
    MaybeCalculateCompleteType ();
    
    lldb::ValueObjectSP
    GetValueForExpressionPath_Impl(const char* expression_cstr,
                                   const char** first_unparsed,
                                   ExpressionPathScanEndReason* reason_to_stop,
                                   ExpressionPathEndResultType* final_value_type,
                                   const GetValueForExpressionPathOptions& options,
                                   ExpressionPathAftermath* final_task_on_target);
        
    // this method will ONLY expand [] expressions into a VOList and return
    // the number of elements it added to the VOList
    // it will NOT loop through expanding the follow-up of the expression_cstr
    // for all objects in the list
    int
    ExpandArraySliceExpression(const char* expression_cstr,
                               const char** first_unparsed,
                               lldb::ValueObjectSP root,
                               lldb::ValueObjectListSP& list,
                               ExpressionPathScanEndReason* reason_to_stop,
                               ExpressionPathEndResultType* final_value_type,
                               const GetValueForExpressionPathOptions& options,
                               ExpressionPathAftermath* final_task_on_target);
                               
    
    DISALLOW_COPY_AND_ASSIGN (ValueObject);

};

} // namespace lldb_private

#endif  // liblldb_ValueObject_h_
