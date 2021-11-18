PyObject *SBTypeToSWIGWrapper(lldb::SBEvent &event_sb) {
  return SWIG_NewPointerObj(&event_sb, SWIGTYPE_p_lldb__SBEvent, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBProcess &process_sb) {
  return SWIG_NewPointerObj(&process_sb, SWIGTYPE_p_lldb__SBProcess, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBThread &thread_sb) {
  return SWIG_NewPointerObj(&thread_sb, SWIGTYPE_p_lldb__SBThread, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBThreadPlan &thread_plan_sb) {
  return SWIG_NewPointerObj(&thread_plan_sb, SWIGTYPE_p_lldb__SBThreadPlan, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBTarget &target_sb) {
  return SWIG_NewPointerObj(&target_sb, SWIGTYPE_p_lldb__SBTarget, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBFrame &frame_sb) {
  return SWIG_NewPointerObj(&frame_sb, SWIGTYPE_p_lldb__SBFrame, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBDebugger &debugger_sb) {
  return SWIG_NewPointerObj(&debugger_sb, SWIGTYPE_p_lldb__SBDebugger, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBBreakpoint &breakpoint_sb) {
  return SWIG_NewPointerObj(&breakpoint_sb, SWIGTYPE_p_lldb__SBBreakpoint, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBWatchpoint &watchpoint_sb) {
  return SWIG_NewPointerObj(&watchpoint_sb, SWIGTYPE_p_lldb__SBWatchpoint, 0);
}

PyObject *
SBTypeToSWIGWrapper(lldb::SBBreakpointLocation &breakpoint_location_sb) {
  return SWIG_NewPointerObj(&breakpoint_location_sb,
                            SWIGTYPE_p_lldb__SBBreakpointLocation, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBValue &value_sb) {
  return SWIG_NewPointerObj(&value_sb, SWIGTYPE_p_lldb__SBValue, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBCommandReturnObject &cmd_ret_obj_sb) {
  return SWIG_NewPointerObj(&cmd_ret_obj_sb,
                            SWIGTYPE_p_lldb__SBCommandReturnObject, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBExecutionContext &ctx_sb) {
  return SWIG_NewPointerObj(&ctx_sb, SWIGTYPE_p_lldb__SBExecutionContext, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBTypeSummaryOptions &summary_options_sb) {
  return SWIG_NewPointerObj(&summary_options_sb,
                            SWIGTYPE_p_lldb__SBTypeSummaryOptions, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBStructuredData &structured_data_sb) {
  return SWIG_NewPointerObj(&structured_data_sb,
                            SWIGTYPE_p_lldb__SBStructuredData, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBSymbolContext &sym_ctx_sb) {
  return SWIG_NewPointerObj(&sym_ctx_sb, SWIGTYPE_p_lldb__SBSymbolContext, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBStream &stream_sb) {
  return SWIG_NewPointerObj(&stream_sb, SWIGTYPE_p_lldb__SBStream, 0);
}
