// leaving this undefined ensures we will get a linker error if we try to use SBTypeToSWIGWrapper()
// for a type for which we did not specialze this function
template <typename SBClass>
PyObject*
SBTypeToSWIGWrapper (SBClass* sb_object);

template <typename SBClass>
PyObject*
SBTypeToSWIGWrapper (SBClass& sb_object)
{
    return SBTypeToSWIGWrapper(&sb_object);
}

template <typename SBClass>
PyObject*
SBTypeToSWIGWrapper (const SBClass& sb_object)
{
    return SBTypeToSWIGWrapper(&sb_object);
}

template <>
PyObject*
SBTypeToSWIGWrapper (PyObject* py_object)
{
    return py_object;
}

template <>
PyObject*
SBTypeToSWIGWrapper (unsigned int* c_int)
{
    if (!c_int)
        return NULL;
    return PyInt_FromLong(*c_int);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBEvent* event_sb)
{
    return SWIG_NewPointerObj((void *) event_sb, SWIGTYPE_p_lldb__SBEvent, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBProcess* process_sb)
{
    return SWIG_NewPointerObj((void *) process_sb, SWIGTYPE_p_lldb__SBProcess, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBThread* thread_sb)
{
    return SWIG_NewPointerObj((void *) thread_sb, SWIGTYPE_p_lldb__SBThread, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBThreadPlan* thread_plan_sb)
{
    return SWIG_NewPointerObj((void *) thread_plan_sb, SWIGTYPE_p_lldb__SBThreadPlan, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBTarget* target_sb)
{
    return SWIG_NewPointerObj((void *) target_sb, SWIGTYPE_p_lldb__SBTarget, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBFrame* frame_sb)
{
    return SWIG_NewPointerObj((void *) frame_sb, SWIGTYPE_p_lldb__SBFrame, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBDebugger* debugger_sb)
{
    return SWIG_NewPointerObj((void *) debugger_sb, SWIGTYPE_p_lldb__SBDebugger, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBBreakpoint* breakpoint_sb)
{
    return SWIG_NewPointerObj((void *) breakpoint_sb, SWIGTYPE_p_lldb__SBBreakpoint, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBWatchpoint* watchpoint_sb)
{
    return SWIG_NewPointerObj((void *) watchpoint_sb, SWIGTYPE_p_lldb__SBWatchpoint, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBBreakpointLocation* breakpoint_location_sb)
{
    return SWIG_NewPointerObj((void *) breakpoint_location_sb, SWIGTYPE_p_lldb__SBBreakpointLocation, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBBreakpointName* breakpoint_name_sb)
{
    return SWIG_NewPointerObj((void *) breakpoint_name_sb, SWIGTYPE_p_lldb__SBBreakpointName, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBValue* value_sb)
{
    return SWIG_NewPointerObj((void *) value_sb, SWIGTYPE_p_lldb__SBValue, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBCommandReturnObject* cmd_ret_obj_sb)
{
    return SWIG_NewPointerObj((void *) cmd_ret_obj_sb, SWIGTYPE_p_lldb__SBCommandReturnObject, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBExecutionContext* ctx_sb)
{
    return SWIG_NewPointerObj((void *) ctx_sb, SWIGTYPE_p_lldb__SBExecutionContext, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBTypeSummaryOptions* summary_options_sb)
{
    return SWIG_NewPointerObj((void *) summary_options_sb, SWIGTYPE_p_lldb__SBTypeSummaryOptions, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBStructuredData* structured_data_sb)
{
    return SWIG_NewPointerObj((void *) structured_data_sb, SWIGTYPE_p_lldb__SBStructuredData, 0);
}

template <>
PyObject*
SBTypeToSWIGWrapper (lldb::SBSymbolContext* sym_ctx_sb)
{
    return SWIG_NewPointerObj((void *) sym_ctx_sb, SWIGTYPE_p_lldb__SBSymbolContext, 0);
}
