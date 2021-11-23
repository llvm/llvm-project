namespace lldb_private {
namespace python {

PyObject *SBTypeToSWIGWrapper(lldb::SBEvent &event_sb) {
  return SWIG_NewPointerObj(&event_sb, SWIGTYPE_p_lldb__SBEvent, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBThread &thread_sb) {
  return SWIG_NewPointerObj(&thread_sb, SWIGTYPE_p_lldb__SBThread, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBFrame &frame_sb) {
  return SWIG_NewPointerObj(&frame_sb, SWIGTYPE_p_lldb__SBFrame, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBDebugger &debugger_sb) {
  return SWIG_NewPointerObj(&debugger_sb, SWIGTYPE_p_lldb__SBDebugger, 0);
}

PyObject *SBTypeToSWIGWrapper(lldb::SBWatchpoint &watchpoint_sb) {
  return SWIG_NewPointerObj(&watchpoint_sb, SWIGTYPE_p_lldb__SBWatchpoint, 0);
}

PyObject *
SBTypeToSWIGWrapper(lldb::SBBreakpointLocation &breakpoint_location_sb) {
  return SWIG_NewPointerObj(&breakpoint_location_sb,
                            SWIGTYPE_p_lldb__SBBreakpointLocation, 0);
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

PythonObject ToSWIGHelper(void *obj, swig_type_info *info) {
  return {PyRefType::Owned, SWIG_NewPointerObj(obj, info, SWIG_POINTER_OWN)};
}

PythonObject ToSWIGWrapper(std::unique_ptr<lldb::SBValue> value_sb) {
  return ToSWIGHelper(value_sb.release(), SWIGTYPE_p_lldb__SBValue);
}

PythonObject ToSWIGWrapper(lldb::ValueObjectSP value_sp) {
  return ToSWIGWrapper(std::make_unique<lldb::SBValue>(std::move(value_sp)));
}

PythonObject ToSWIGWrapper(lldb::TargetSP target_sp) {
  return ToSWIGHelper(new lldb::SBTarget(std::move(target_sp)),
                      SWIGTYPE_p_lldb__SBTarget);
}

PythonObject ToSWIGWrapper(lldb::ProcessSP process_sp) {
  return ToSWIGHelper(new lldb::SBProcess(std::move(process_sp)),
                      SWIGTYPE_p_lldb__SBProcess);
}

PythonObject ToSWIGWrapper(lldb::ThreadPlanSP thread_plan_sp) {
  return ToSWIGHelper(new lldb::SBThreadPlan(std::move(thread_plan_sp)),
                      SWIGTYPE_p_lldb__SBThreadPlan);
}

PythonObject ToSWIGWrapper(lldb::BreakpointSP breakpoint_sp) {
  return ToSWIGHelper(new lldb::SBBreakpoint(std::move(breakpoint_sp)),
                      SWIGTYPE_p_lldb__SBBreakpoint);
}

} // namespace python
} // namespace lldb_private
