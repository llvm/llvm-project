%header %{

class PyErr_Cleaner {
public:
  PyErr_Cleaner(bool print = false) : m_print(print) {}

  ~PyErr_Cleaner() {
    if (PyErr_Occurred()) {
      if (m_print && !PyErr_ExceptionMatches(PyExc_SystemExit))
        PyErr_Print();
      PyErr_Clear();
    }
  }

private:
  bool m_print;
};

llvm::Expected<bool> lldb_private::python::SWIGBridge::LLDBSwigPythonBreakpointCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::StackFrameSP &frame_sp,
    const lldb::BreakpointLocationSP &bp_loc_sp,
    const lldb_private::StructuredDataImpl &args_impl) {
  using namespace llvm;

  lldb::SBBreakpointLocation sb_bp_loc(bp_loc_sp);

  PyErr_Cleaner py_err_cleaner(true);
  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  unsigned max_positional_args;
  if (auto arg_info = pfunc.GetArgInfo())
    max_positional_args = arg_info.get().max_positional_args;
  else
    return arg_info.takeError();

  PythonObject frame_arg = SWIGBridge::ToSWIGWrapper(frame_sp);
  PythonObject bp_loc_arg = SWIGBridge::ToSWIGWrapper(bp_loc_sp);

  auto result =
      max_positional_args < 4
          ? pfunc.Call(frame_arg, bp_loc_arg, dict)
          : pfunc.Call(frame_arg, bp_loc_arg, SWIGBridge::ToSWIGWrapper(args_impl), dict);

  if (!result)
    return result.takeError();

  // Only False counts as false!
  return result.get().get() != Py_False;
}

// resolve a dotted Python name in the form
// foo.bar.baz.Foobar to an actual Python object
// if pmodule is NULL, the __main__ module will be used
// as the starting point for the search

// This function is called by
// lldb_private::ScriptInterpreterPython::BreakpointCallbackFunction(...) and is
// used when a script command is attached to a breakpoint for execution.

// This function is called by
// lldb_private::ScriptInterpreterPython::WatchpointCallbackFunction(...) and is
// used when a script command is attached to a watchpoint for execution.

bool lldb_private::python::SWIGBridge::LLDBSwigPythonWatchpointCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::StackFrameSP &frame_sp, const lldb::WatchpointSP &wp_sp) {

  bool stop_at_watchpoint = true;

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return stop_at_watchpoint;

  PythonObject result =
      pfunc(SWIGBridge::ToSWIGWrapper(frame_sp), SWIGBridge::ToSWIGWrapper(wp_sp), dict);

  if (result.get() == Py_False)
    stop_at_watchpoint = false;

  return stop_at_watchpoint;
}

// This function is called by
// ScriptInterpreterPython::FormatterMatchingCallbackFunction and it's used when
// a data formatter provides the name of a callback to inspect a candidate type
// before considering a match.
bool lldb_private::python::SWIGBridge::LLDBSwigPythonFormatterCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::TypeImplSP type_impl_sp) {

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return false;

  PythonObject result =
      pfunc(SWIGBridge::ToSWIGWrapper(type_impl_sp), dict);

  // Only if everything goes okay and the function returns True we'll consider
  // it a match.
  return result.get() == Py_True;
}

bool lldb_private::python::SWIGBridge::LLDBSwigPythonCallTypeScript(
    const char *python_function_name, const void *session_dictionary,
    const lldb::ValueObjectSP &valobj_sp, void **pyfunct_wrapper,
    const lldb::TypeSummaryOptionsSP &options_sp, std::string &retval) {

  retval.clear();

  if (!python_function_name || !session_dictionary)
    return false;

  PyObject *pfunc_impl = nullptr;

  if (pyfunct_wrapper && *pyfunct_wrapper
#ifndef Py_LIMITED_API
      && PyFunction_Check(*pyfunct_wrapper)
#endif
      ) {
    pfunc_impl = (PyObject *)(*pyfunct_wrapper);
    if (Py_REFCNT(pfunc_impl) == 1) {
      Py_XDECREF(pfunc_impl);
      pfunc_impl = NULL;
    }
  }

  PyObject *py_dict = (PyObject *)session_dictionary;
  if (!PythonDictionary::Check(py_dict))
    return true;

  PythonDictionary dict(PyRefType::Borrowed, py_dict);

  PyErr_Cleaner pyerr_cleanup(true); // show Python errors

  PythonCallable pfunc(PyRefType::Borrowed, pfunc_impl);

  if (!pfunc.IsAllocated()) {
    pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
        python_function_name, dict);
    if (!pfunc.IsAllocated())
      return false;

    if (pyfunct_wrapper) {
      *pyfunct_wrapper = pfunc.get();
      Py_XINCREF(pfunc.get());
    }
  }

  PythonObject result;
  auto argc = pfunc.GetArgInfo();
  if (!argc) {
    llvm::consumeError(argc.takeError());
    return false;
  }

  PythonObject value_arg = SWIGBridge::ToSWIGWrapper(valobj_sp);

  if (argc.get().max_positional_args < 3)
    result = pfunc(value_arg, dict);
  else
    result = pfunc(value_arg, dict, SWIGBridge::ToSWIGWrapper(*options_sp));

  retval = result.Str().GetString().str();

  return true;
}

PythonObject lldb_private::python::SWIGBridge::LLDBSwigPythonCreateCommandObject(
    const char *python_class_name, const char *session_dictionary_name,
    lldb::DebuggerSP debugger_sp) {
  if (python_class_name == NULL || python_class_name[0] == '\0' ||
      !session_dictionary_name)
    return PythonObject();

  PyErr_Cleaner py_err_cleaner(true);
  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_class_name, dict);

  if (!pfunc.IsAllocated())
    return PythonObject();

  return pfunc(SWIGBridge::ToSWIGWrapper(std::move(debugger_sp)), dict);
}

// wrapper that calls an optional instance member of an object taking no
// arguments
static PyObject *LLDBSwigPython_CallOptionalMember(
    PyObject * implementor, char *callee_name,
    PyObject *ret_if_not_found = Py_None, bool *was_found = NULL) {
  PyErr_Cleaner py_err_cleaner(false);

  PythonObject self(PyRefType::Borrowed, static_cast<PyObject *>(implementor));
  auto pfunc = self.ResolveName<PythonCallable>(callee_name);

  if (!pfunc.IsAllocated()) {
    if (was_found)
      *was_found = false;
    Py_XINCREF(ret_if_not_found);
    return ret_if_not_found;
  }

  if (was_found)
    *was_found = true;

  PythonObject result = pfunc();
  return result.release();
}


void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBData(PyObject * data) {
  lldb::SBData *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBData, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBThreadPlan(PyObject * data) {
  lldb::SBThreadPlan *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBThreadPlan, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBBreakpoint(PyObject * data) {
  lldb::SBBreakpoint *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBBreakpoint, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBThread(PyObject * data) {
  lldb::SBThread *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBThread, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBFrame(PyObject * data) {
  lldb::SBFrame *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBFrame, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBBreakpointLocation(PyObject * data) {
  lldb::SBBreakpointLocation *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBBreakpointLocation, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBAttachInfo(PyObject * data) {
  lldb::SBAttachInfo *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBAttachInfo, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBLaunchInfo(PyObject * data) {
  lldb::SBLaunchInfo *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBLaunchInfo, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBError(PyObject * data) {
  lldb::SBError *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBError, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBCommandReturnObject(
    PyObject *data) {
  lldb::SBCommandReturnObject *sb_ptr = nullptr;

  int valid_cast = SWIG_ConvertPtr(
      data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBCommandReturnObject, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBDebugger(
    PyObject *data) {
  lldb::SBDebugger *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBDebugger, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBEvent(PyObject * data) {
  lldb::SBEvent *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBEvent, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBStream(PyObject * data) {
  lldb::SBStream *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBStream, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBSymbolContext(PyObject * data) {
  lldb::SBSymbolContext *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBSymbolContext, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBValue(PyObject * data) {
  lldb::SBValue *sb_ptr = NULL;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBValue, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBValueList(PyObject * data) {
  lldb::SBValueList *sb_ptr = NULL;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBValueList, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBMemoryRegionInfo(PyObject *
                                                                    data) {
  lldb::SBMemoryRegionInfo *sb_ptr = NULL;

  int valid_cast = SWIG_ConvertPtr(data, (void **)&sb_ptr,
                                   SWIGTYPE_p_lldb__SBMemoryRegionInfo, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBExecutionContext(PyObject *
                                                                    data) {
  lldb::SBExecutionContext *sb_ptr = NULL;

  int valid_cast = SWIG_ConvertPtr(data, (void **)&sb_ptr,
                                   SWIGTYPE_p_lldb__SBExecutionContext, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBFrameList(PyObject *data) {
  lldb::SBFrameList *sb_ptr = NULL;

  int valid_cast = SWIG_ConvertPtr(data, (void **)&sb_ptr,
                                   SWIGTYPE_p_lldb__SBFrameList, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::python::LLDBSWIGPython_CastPyObjectToSBTarget(PyObject *
                                                                    data) {
  lldb::SBTarget *sb_ptr = NULL;

  int valid_cast = SWIG_ConvertPtr(data, (void **)&sb_ptr,
                                   SWIGTYPE_p_lldb__SBTarget, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;                                                                                                                                                                                  
}                                                                                                                                                                                                 
bool lldb_private::python::SWIGBridge::LLDBSwigPythonCallCommand(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::DebuggerSP debugger, const char *args,
    lldb_private::CommandReturnObject &cmd_retobj,
    lldb::ExecutionContextRefSP exe_ctx_ref_sp) {

  PyErr_Cleaner py_err_cleaner(true);
  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return false;

  auto argc = pfunc.GetArgInfo();
  if (!argc) {
    llvm::consumeError(argc.takeError());
    return false;
  }
  PythonObject debugger_arg = SWIGBridge::ToSWIGWrapper(std::move(debugger));
  auto cmd_retobj_arg = SWIGBridge::ToSWIGWrapper(cmd_retobj);

  if (argc.get().max_positional_args < 5u)
    pfunc(debugger_arg, PythonString(args), cmd_retobj_arg.obj(), dict);
  else
    pfunc(debugger_arg, PythonString(args),
          SWIGBridge::ToSWIGWrapper(std::move(exe_ctx_ref_sp)), cmd_retobj_arg.obj(), dict);

  // `pfunc`'s bare call syntax doesn't drain the interpreter's exception
  // state, so check it directly: a Python runtime error raised from inside
  // the command body must be reported as a failure (surfaced through
  // `cmd_retobj`'s error status), not silently treated as success.
  return !PyErr_Occurred();
}

PythonObject lldb_private::python::SWIGBridge::LLDBSWIGPythonCreateOSPlugin(
    const char *python_class_name, const char *session_dictionary_name,
    const lldb::ProcessSP &process_sp) {
  if (python_class_name == NULL || python_class_name[0] == '\0' ||
      !session_dictionary_name)
    return PythonObject();

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_class_name, dict);

  if (!pfunc.IsAllocated())
    return PythonObject();

  return pfunc(SWIGBridge::ToSWIGWrapper(process_sp));
}

void *lldb_private::python::SWIGBridge::LLDBSWIGPython_GetDynamicSetting(
    void *module, const char *setting, const lldb::TargetSP &target_sp) {
  if (!module || !setting)
    Py_RETURN_NONE;

  PyErr_Cleaner py_err_cleaner(true);
  PythonObject py_module(PyRefType::Borrowed, (PyObject *)module);
  auto pfunc = py_module.ResolveName<PythonCallable>("get_dynamic_setting");

  if (!pfunc.IsAllocated())
    Py_RETURN_NONE;

  auto result = pfunc(SWIGBridge::ToSWIGWrapper(target_sp), PythonString(setting));

  return result.release();
}

bool lldb_private::python::SWIGBridge::LLDBSWIGPythonRunScriptKeywordProcess(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::ProcessSP &process, std::string &output) {

  if (python_function_name == NULL || python_function_name[0] == '\0' ||
      !session_dictionary_name)
    return false;

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return false;

  auto result = pfunc(SWIGBridge::ToSWIGWrapper(process), dict);

  output = result.Str().GetString().str();

  return true;
}

std::optional<std::string> lldb_private::python::SWIGBridge::LLDBSWIGPythonRunScriptKeywordThread(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ThreadSP thread) {
  if (python_function_name == NULL || python_function_name[0] == '\0' ||
      !session_dictionary_name)
    return std::nullopt;

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return std::nullopt;

  auto result = pfunc(SWIGBridge::ToSWIGWrapper(std::move(thread)), dict);

  return result.Str().GetString().str();
}

bool lldb_private::python::SWIGBridge::LLDBSWIGPythonRunScriptKeywordTarget(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::TargetSP &target, std::string &output) {

  if (python_function_name == NULL || python_function_name[0] == '\0' ||
      !session_dictionary_name)
    return false;

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return false;

  auto result = pfunc(SWIGBridge::ToSWIGWrapper(target), dict);

  output = result.Str().GetString().str();

  return true;
}

std::optional<std::string> lldb_private::python::SWIGBridge::LLDBSWIGPythonRunScriptKeywordFrame(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::StackFrameSP frame) {
  if (python_function_name == NULL || python_function_name[0] == '\0' ||
      !session_dictionary_name)
    return std::nullopt;

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return std::nullopt;

  auto result = pfunc(SWIGBridge::ToSWIGWrapper(std::move(frame)), dict);

  return result.Str().GetString().str();
}

bool lldb_private::python::SWIGBridge::LLDBSWIGPythonRunScriptKeywordValue(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::ValueObjectSP &value, std::string &output) {

  if (python_function_name == NULL || python_function_name[0] == '\0' ||
      !session_dictionary_name)
    return false;

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return false;

  auto result = pfunc(SWIGBridge::ToSWIGWrapper(value), dict);

  output = result.Str().GetString().str();

  return true;
}

bool lldb_private::python::SWIGBridge::LLDBSwigPythonCallModuleNewTarget(
    const char *python_module_name, const char *session_dictionary_name,
    lldb::TargetSP target_sp) {
  std::string python_function_name_string = python_module_name;
  python_function_name_string += ".__lldb_module_added_to_target";
  const char *python_function_name = python_function_name_string.c_str();

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return true;

  pfunc(SWIGBridge::ToSWIGWrapper(std::move(target_sp)), dict);

  // `pfunc`'s bare call syntax doesn't drain the interpreter's exception
  // state, so check it directly: an exception raised by
  // `__lldb_module_added_to_target` must be reported as a failure, not
  // silently treated as success.
  return !PyErr_Occurred();
}

bool lldb_private::python::SWIGBridge::LLDBSwigPythonCallModuleInit(
    const char *python_module_name, const char *session_dictionary_name,
    lldb::DebuggerSP debugger) {
  std::string python_function_name_string = python_module_name;
  python_function_name_string += ".__lldb_init_module";
  const char *python_function_name = python_function_name_string.c_str();

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  // This method is optional and need not exist.  So if we don't find it,
  // it's actually a success, not a failure.
  if (!pfunc.IsAllocated())
    return true;

  pfunc(SWIGBridge::ToSWIGWrapper(std::move(debugger)), dict);

  // Same as above: an exception raised by __lldb_init_module must be
  // reported as a failure so callers (e.g. `command script import`, or
  // module auto-loading) don't treat it as a successful import.
  return !PyErr_Occurred();
}

lldb::ValueObjectSP lldb_private::python::SWIGBridge::LLDBSWIGPython_GetValueObjectSPFromSBValue(
    void *data) {
  lldb::ValueObjectSP valobj_sp;
  if (data) {
    lldb::SBValue *sb_ptr = (lldb::SBValue *)data;
    valobj_sp = sb_ptr->GetSP();
  }
  return valobj_sp;
}

// For the LogOutputCallback functions
static void LLDBSwigPythonCallPythonLogOutputCallback(const char *str,
                                                      void *baton) {
  if (baton != Py_None) {
    SWIG_PYTHON_THREAD_BEGIN_BLOCK;
    PyObject *result = PyObject_CallFunction(
        reinterpret_cast<PyObject *>(baton), const_cast<char *>("s"), str);
    Py_XDECREF(result);
    SWIG_PYTHON_THREAD_END_BLOCK;
  }
}

// For CommandPrintCallback functions
static CommandReturnObjectCallbackResult LLDBSwigPythonCallPythonCommandPrintCallback(SBCommandReturnObject& result, void *callback_baton) {
  SWIG_Python_Thread_Block swig_thread_block;

  PyErr_Cleaner py_err_cleaner(true);

  PythonObject result_arg = SWIGBridge::ToSWIGWrapper(
      std::make_unique<SBCommandReturnObject>(result));
  PythonCallable callable =
      Retain<PythonCallable>(reinterpret_cast<PyObject *>(callback_baton));

  if (!callable.IsValid())
    return eCommandReturnObjectPrintCallbackSkipped;

  PythonObject callback_result = callable(result_arg);

  long long ret_val = unwrapOrSetPythonException(As<long long>(callback_result));
  return (CommandReturnObjectCallbackResult)ret_val;
}

// For DebuggerTerminateCallback functions
static void LLDBSwigPythonCallPythonSBDebuggerTerminateCallback(lldb::user_id_t debugger_id,
                                                      void *baton) {
  if (baton != Py_None) {
    SWIG_PYTHON_THREAD_BEGIN_BLOCK;
    PyObject *result = PyObject_CallFunction(
        reinterpret_cast<PyObject *>(baton), const_cast<char *>("l"), debugger_id);
    Py_XDECREF(result);
    SWIG_PYTHON_THREAD_END_BLOCK;
  }
}

static bool LLDBSwigPythonCallPythonSBCommandInterpreterSetCommandOverrideCallback(void *baton, const char **argv) {
  bool ret_val = false;
  if (baton != Py_None) {
    SWIG_PYTHON_THREAD_BEGIN_BLOCK;
    // Create a PyList of items since we're going to pass it to the callback as a tuple
    // of arguments.
    PyObject *py_argv = PyList_New(0);
    for (const char **arg = argv; arg && *arg; arg++) {
      std::string arg_string = *arg;
      PyObject *py_string = PyUnicode_FromStringAndSize(arg_string.c_str(), arg_string.size());
      PyList_Append(py_argv, py_string);
    }

    PyObject *result = PyObject_CallObject(
        reinterpret_cast<PyObject *>(baton), PyList_AsTuple(py_argv));
    ret_val = result ? PyObject_IsTrue(result) : false;
    Py_XDECREF(result);
    SWIG_PYTHON_THREAD_END_BLOCK;
  }
  return ret_val;
}

static SBError LLDBSwigPythonCallLocateModuleCallback(
    void *callback_baton, const SBModuleSpec &module_spec_sb,
    SBFileSpec &module_file_spec_sb, SBFileSpec &symbol_file_spec_sb) {
  SWIG_Python_Thread_Block swig_thread_block;

  PyErr_Cleaner py_err_cleaner(true);
  PythonObject module_spec_arg = SWIGBridge::ToSWIGWrapper(
      std::make_unique<SBModuleSpec>(module_spec_sb));
  PythonObject module_file_spec_arg = SWIGBridge::ToSWIGWrapper(
      std::make_unique<SBFileSpec>(module_file_spec_sb));
  PythonObject symbol_file_spec_arg = SWIGBridge::ToSWIGWrapper(
      std::make_unique<SBFileSpec>(symbol_file_spec_sb));

  PythonCallable callable =
      Retain<PythonCallable>(reinterpret_cast<PyObject *>(callback_baton));
  if (!callable.IsValid()) {
    return SBError("The callback callable is not valid.");
  }

  PythonObject result = callable(module_spec_arg, module_file_spec_arg,
                                 symbol_file_spec_arg);

  if (!result.IsAllocated())
    return SBError("No result.");
  lldb::SBError *sb_error_ptr = nullptr;
  if (SWIG_ConvertPtr(result.get(), (void **)&sb_error_ptr,
                      SWIGTYPE_p_lldb__SBError, 0) == -1) {
    return SBError("Result is not SBError.");
  }

  if (sb_error_ptr->Success()) {
    lldb::SBFileSpec *sb_module_file_spec_ptr = nullptr;
    if (SWIG_ConvertPtr(module_file_spec_arg.get(),
                        (void **)&sb_module_file_spec_ptr,
                        SWIGTYPE_p_lldb__SBFileSpec, 0) == -1)
      return SBError("module_file_spec is not SBFileSpec.");

    lldb::SBFileSpec *sb_symbol_file_spec_ptr = nullptr;
    if (SWIG_ConvertPtr(symbol_file_spec_arg.get(),
                        (void **)&sb_symbol_file_spec_ptr,
                        SWIGTYPE_p_lldb__SBFileSpec, 0) == -1)
      return SBError("symbol_file_spec is not SBFileSpec.");

    module_file_spec_sb = *sb_module_file_spec_ptr;
    symbol_file_spec_sb = *sb_symbol_file_spec_ptr;
  }

  return *sb_error_ptr;
}
%}
