//===-- StackFrame.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StackFrame_h_
#define liblldb_StackFrame_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackID.h"

namespace lldb_private {

class StackFrame :
    public std::enable_shared_from_this<StackFrame>,
    public ExecutionContextScope
{
public:
    enum ExpressionPathOption
    {
        eExpressionPathOptionCheckPtrVsMember       = (1u << 0),
        eExpressionPathOptionsNoFragileObjcIvar     = (1u << 1),
        eExpressionPathOptionsNoSyntheticChildren   = (1u << 2),
        eExpressionPathOptionsNoSyntheticArrayRange = (1u << 3),
        eExpressionPathOptionsAllowDirectIVarAccess = (1u << 4)
    };
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    StackFrame (const lldb::ThreadSP &thread_sp,
                lldb::user_id_t frame_idx, 
                lldb::user_id_t concrete_frame_idx, 
                lldb::addr_t cfa, 
                lldb::addr_t pc, 
                const SymbolContext *sc_ptr);

    StackFrame (const lldb::ThreadSP &thread_sp,
                lldb::user_id_t frame_idx, 
                lldb::user_id_t concrete_frame_idx, 
                const lldb::RegisterContextSP &reg_context_sp, 
                lldb::addr_t cfa, 
                lldb::addr_t pc, 
                const SymbolContext *sc_ptr);
    
    StackFrame (const lldb::ThreadSP &thread_sp,
                lldb::user_id_t frame_idx, 
                lldb::user_id_t concrete_frame_idx, 
                const lldb::RegisterContextSP &reg_context_sp, 
                lldb::addr_t cfa, 
                const Address& pc, 
                const SymbolContext *sc_ptr);

    virtual ~StackFrame ();

    lldb::ThreadSP
    GetThread () const
    {
        return m_thread_wp.lock();
    }

    StackID&
    GetStackID();

    const Address&
    GetFrameCodeAddress();
    
    void
    ChangePC (lldb::addr_t pc);

    const SymbolContext&
    GetSymbolContext (uint32_t resolve_scope);

    bool
    GetFrameBaseValue(Scalar &value, Error *error_ptr);

    Block *
    GetFrameBlock ();

    lldb::RegisterContextSP
    GetRegisterContext ();

    const lldb::RegisterContextSP &
    GetRegisterContextSP () const
    {
        return m_reg_context_sp;
    }

    VariableList *
    GetVariableList (bool get_file_globals);

    lldb::VariableListSP
    GetInScopeVariableList (bool get_file_globals);

    // See ExpressionPathOption enumeration for "options" values
    lldb::ValueObjectSP
    GetValueForVariableExpressionPath (const char *var_expr, 
                                       lldb::DynamicValueType use_dynamic, 
                                       uint32_t options,
                                       lldb::VariableSP &var_sp,
                                       Error &error);

    bool
    HasDebugInformation ();

    const char *
    Disassemble ();

    void
    DumpUsingSettingsFormat (Stream *strm);
    
    void
    Dump (Stream *strm, bool show_frame_index, bool show_fullpaths);
    
    bool
    IsInlined ();

    uint32_t
    GetFrameIndex () const;

    uint32_t
    GetConcreteFrameIndex () const
    {
        return m_concrete_frame_index;
    }
    
    lldb::ValueObjectSP
    GetValueObjectForFrameVariable (const lldb::VariableSP &variable_sp, lldb::DynamicValueType use_dynamic);

    lldb::ValueObjectSP
    TrackGlobalVariable (const lldb::VariableSP &variable_sp, lldb::DynamicValueType use_dynamic);
    
    //------------------------------------------------------------------
    // lldb::ExecutionContextScope pure virtual functions
    //------------------------------------------------------------------
    virtual lldb::TargetSP
    CalculateTarget ();
    
    virtual lldb::ProcessSP
    CalculateProcess ();
    
    virtual lldb::ThreadSP
    CalculateThread ();
    
    virtual lldb::StackFrameSP
    CalculateStackFrame ();

    virtual void
    CalculateExecutionContext (ExecutionContext &exe_ctx);
    
    bool
    GetStatus (Stream &strm,
               bool show_frame_info,
               bool show_source);
    
protected:
    friend class StackFrameList;

    void
    SetSymbolContextScope (SymbolContextScope *symbol_scope);

    void
    UpdateCurrentFrameFromPreviousFrame (StackFrame &prev_frame);
    
    void
    UpdatePreviousFrameFromCurrentFrame (StackFrame &curr_frame);

    bool
    HasCachedData () const;
    
private:
    //------------------------------------------------------------------
    // For StackFrame only
    //------------------------------------------------------------------
    lldb::ThreadWP m_thread_wp;
    uint32_t m_frame_index;
    uint32_t m_concrete_frame_index;
    lldb::RegisterContextSP m_reg_context_sp;
    StackID m_id;
    Address m_frame_code_addr;   // The frame code address (might not be the same as the actual PC for inlined frames) as a section/offset address
    SymbolContext m_sc;
    Flags m_flags;
    Scalar m_frame_base;
    Error m_frame_base_error;
    lldb::VariableListSP m_variable_list_sp;
    ValueObjectList m_variable_list_value_objects;  // Value objects for each variable in m_variable_list_sp
    StreamString m_disassembly;
    DISALLOW_COPY_AND_ASSIGN (StackFrame);
};

} // namespace lldb_private

#endif  // liblldb_StackFrame_h_
