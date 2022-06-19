namespace lldb_private {
namespace python {

PythonObject ToSWIGHelper(void *obj, swig_type_info *info) {
  return {PyRefType::Owned, SWIG_NewPointerObj(obj, info, SWIG_POINTER_OWN)};
}

/// A class that automatically clears an SB object when it goes out of scope.
/// Use for cases where the SB object points to a temporary/unowned entity.
template <typename T> class ScopedPythonObject : PythonObject {
public:
  ScopedPythonObject(T *sb, swig_type_info *info)
      : PythonObject(ToSWIGHelper(sb, info)), m_sb(sb) {}
  ~ScopedPythonObject() {
    if (m_sb)
      *m_sb = T();
  }
  ScopedPythonObject(ScopedPythonObject &&rhs)
      : PythonObject(std::move(rhs)), m_sb(std::exchange(rhs.m_sb, nullptr)) {}
  ScopedPythonObject(const ScopedPythonObject &) = delete;
  ScopedPythonObject &operator=(const ScopedPythonObject &) = delete;
  ScopedPythonObject &operator=(ScopedPythonObject &&) = delete;

  const PythonObject &obj() const { return *this; }

private:
  T *m_sb;
};

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

PythonObject ToSWIGWrapper(std::unique_ptr<lldb::SBStream> stream_sb) {
  return ToSWIGHelper(stream_sb.release(), SWIGTYPE_p_lldb__SBStream);
}

PythonObject ToSWIGWrapper(std::unique_ptr<lldb::SBStructuredData> data_sb) {
  return ToSWIGHelper(data_sb.release(), SWIGTYPE_p_lldb__SBStructuredData);
}

PythonObject ToSWIGWrapper(const StructuredDataImpl &data_impl) {
  return ToSWIGWrapper(std::make_unique<lldb::SBStructuredData>(data_impl));
}

PythonObject ToSWIGWrapper(lldb::ThreadSP thread_sp) {
  return ToSWIGHelper(new lldb::SBThread(std::move(thread_sp)),
                      SWIGTYPE_p_lldb__SBThread);
}

PythonObject ToSWIGWrapper(lldb::StackFrameSP frame_sp) {
  return ToSWIGHelper(new lldb::SBFrame(std::move(frame_sp)),
                      SWIGTYPE_p_lldb__SBFrame);
}

PythonObject ToSWIGWrapper(lldb::DebuggerSP debugger_sp) {
  return ToSWIGHelper(new lldb::SBDebugger(std::move(debugger_sp)),
                      SWIGTYPE_p_lldb__SBDebugger);
}

PythonObject ToSWIGWrapper(lldb::WatchpointSP watchpoint_sp) {
  return ToSWIGHelper(new lldb::SBWatchpoint(std::move(watchpoint_sp)),
                      SWIGTYPE_p_lldb__SBWatchpoint);
}

PythonObject ToSWIGWrapper(lldb::BreakpointLocationSP bp_loc_sp) {
  return ToSWIGHelper(new lldb::SBBreakpointLocation(std::move(bp_loc_sp)),
                      SWIGTYPE_p_lldb__SBBreakpointLocation);
}

PythonObject ToSWIGWrapper(lldb::ExecutionContextRefSP ctx_sp) {
  return ToSWIGHelper(new lldb::SBExecutionContext(std::move(ctx_sp)),
                      SWIGTYPE_p_lldb__SBExecutionContext);
}

PythonObject ToSWIGWrapper(const TypeSummaryOptions &summary_options) {
  return ToSWIGHelper(new lldb::SBTypeSummaryOptions(summary_options),
                      SWIGTYPE_p_lldb__SBTypeSummaryOptions);
}

PythonObject ToSWIGWrapper(const SymbolContext &sym_ctx) {
  return ToSWIGHelper(new lldb::SBSymbolContext(sym_ctx),
                      SWIGTYPE_p_lldb__SBSymbolContext);
}

ScopedPythonObject<lldb::SBCommandReturnObject>
ToSWIGWrapper(CommandReturnObject &cmd_retobj) {
  return ScopedPythonObject<lldb::SBCommandReturnObject>(
      new lldb::SBCommandReturnObject(cmd_retobj),
      SWIGTYPE_p_lldb__SBCommandReturnObject);
}

ScopedPythonObject<lldb::SBEvent> ToSWIGWrapper(Event *event) {
  return ScopedPythonObject<lldb::SBEvent>(new lldb::SBEvent(event),
                                           SWIGTYPE_p_lldb__SBEvent);
}

} // namespace python
} // namespace lldb_private
