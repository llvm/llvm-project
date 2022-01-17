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

llvm::Expected<bool> lldb_private::LLDBSwigPythonBreakpointCallbackFunction(
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

  PythonObject frame_arg = ToSWIGWrapper(frame_sp);
  PythonObject bp_loc_arg = ToSWIGWrapper(bp_loc_sp);

  auto result =
      max_positional_args < 4
          ? pfunc.Call(frame_arg, bp_loc_arg, dict)
          : pfunc.Call(frame_arg, bp_loc_arg, ToSWIGWrapper(args_impl), dict);

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

bool lldb_private::LLDBSwigPythonWatchpointCallbackFunction(
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
      pfunc(ToSWIGWrapper(frame_sp), ToSWIGWrapper(wp_sp), dict);

  if (result.get() == Py_False)
    stop_at_watchpoint = false;

  return stop_at_watchpoint;
}

bool lldb_private::LLDBSwigPythonCallTypeScript(
    const char *python_function_name, const void *session_dictionary,
    const lldb::ValueObjectSP &valobj_sp, void **pyfunct_wrapper,
    const lldb::TypeSummaryOptionsSP &options_sp, std::string &retval) {

  retval.clear();

  if (!python_function_name || !session_dictionary)
    return false;

  PyObject *pfunc_impl = nullptr;

  if (pyfunct_wrapper && *pyfunct_wrapper &&
      PyFunction_Check(*pyfunct_wrapper)) {
    pfunc_impl = (PyObject *)(*pyfunct_wrapper);
    if (pfunc_impl->ob_refcnt == 1) {
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

  PythonObject value_arg = ToSWIGWrapper(valobj_sp);

  if (argc.get().max_positional_args < 3)
    result = pfunc(value_arg, dict);
  else
    result = pfunc(value_arg, dict, ToSWIGWrapper(*options_sp));

  retval = result.Str().GetString().str();

  return true;
}

PythonObject lldb_private::LLDBSwigPythonCreateSyntheticProvider(
    const char *python_class_name, const char *session_dictionary_name,
    const lldb::ValueObjectSP &valobj_sp) {
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

  auto sb_value = std::make_unique<lldb::SBValue>(valobj_sp);
  sb_value->SetPreferSyntheticValue(false);

  PythonObject val_arg = ToSWIGWrapper(std::move(sb_value));
  if (!val_arg.IsAllocated())
    return PythonObject();

  PythonObject result = pfunc(val_arg, dict);

  if (result.IsAllocated())
    return result;

  return PythonObject();
}

PythonObject lldb_private::LLDBSwigPythonCreateCommandObject(
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

  return pfunc(ToSWIGWrapper(std::move(debugger_sp)), dict);
}

PythonObject lldb_private::LLDBSwigPythonCreateScriptedProcess(
    const char *python_class_name, const char *session_dictionary_name,
    const lldb::TargetSP &target_sp,
    const lldb_private::StructuredDataImpl &args_impl,
    std::string &error_string) {
  if (python_class_name == NULL || python_class_name[0] == '\0' ||
      !session_dictionary_name)
    return PythonObject();

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_class_name, dict);

  if (!pfunc.IsAllocated()) {
    error_string.append("could not find script class: ");
    error_string.append(python_class_name);
    return PythonObject();
  }

  PythonObject target_arg = ToSWIGWrapper(target_sp);

  llvm::Expected<PythonCallable::ArgInfo> arg_info = pfunc.GetArgInfo();
  if (!arg_info) {
    llvm::handleAllErrors(
        arg_info.takeError(),
        [&](PythonException &E) { error_string.append(E.ReadBacktrace()); },
        [&](const llvm::ErrorInfoBase &E) {
          error_string.append(E.message());
        });
    return PythonObject();
  }

  PythonObject result = {};
  if (arg_info.get().max_positional_args == 2) {
    result = pfunc(target_arg, ToSWIGWrapper(args_impl));
  } else {
    error_string.assign("wrong number of arguments in __init__, should be 2 "
                        "(not including self)");
  }
  return result;
}

PythonObject lldb_private::LLDBSwigPythonCreateScriptedThread(
    const char *python_class_name, const char *session_dictionary_name,
    const lldb::ProcessSP &process_sp, const StructuredDataImpl &args_impl,
    std::string &error_string) {
  if (python_class_name == NULL || python_class_name[0] == '\0' ||
      !session_dictionary_name)
    return PythonObject();

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_class_name, dict);

  if (!pfunc.IsAllocated()) {
    error_string.append("could not find script class: ");
    error_string.append(python_class_name);
    return PythonObject();
  }

  llvm::Expected<PythonCallable::ArgInfo> arg_info = pfunc.GetArgInfo();
  if (!arg_info) {
    llvm::handleAllErrors(
        arg_info.takeError(),
        [&](PythonException &E) { error_string.append(E.ReadBacktrace()); },
        [&](const llvm::ErrorInfoBase &E) {
          error_string.append(E.message());
        });
    return PythonObject();
  }

  if (arg_info.get().max_positional_args == 2)
    return pfunc(ToSWIGWrapper(process_sp), ToSWIGWrapper(args_impl));

  error_string.assign("wrong number of arguments in __init__, should be 2 "
                      "(not including self)");
  return PythonObject();
}

PythonObject lldb_private::LLDBSwigPythonCreateScriptedThreadPlan(
    const char *python_class_name, const char *session_dictionary_name,
    const lldb_private::StructuredDataImpl &args_impl,
    std::string &error_string, const lldb::ThreadPlanSP &thread_plan_sp) {
  if (python_class_name == NULL || python_class_name[0] == '\0' ||
      !session_dictionary_name)
    return PythonObject();

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_class_name, dict);

  if (!pfunc.IsAllocated()) {
    error_string.append("could not find script class: ");
    error_string.append(python_class_name);
    return PythonObject();
  }

  PythonObject tp_arg = ToSWIGWrapper(thread_plan_sp);

  llvm::Expected<PythonCallable::ArgInfo> arg_info = pfunc.GetArgInfo();
  if (!arg_info) {
    llvm::handleAllErrors(
        arg_info.takeError(),
        [&](PythonException &E) { error_string.append(E.ReadBacktrace()); },
        [&](const llvm::ErrorInfoBase &E) {
          error_string.append(E.message());
        });
    return PythonObject();
  }

  PythonObject result = {};
  auto args_sb = std::make_unique<lldb::SBStructuredData>(args_impl);
  if (arg_info.get().max_positional_args == 2) {
    if (args_sb->IsValid()) {
      error_string.assign(
          "args passed, but __init__ does not take an args dictionary");
      return PythonObject();
    }
    result = pfunc(tp_arg, dict);
  } else if (arg_info.get().max_positional_args >= 3) {
    result = pfunc(tp_arg, ToSWIGWrapper(std::move(args_sb)), dict);
  } else {
    error_string.assign("wrong number of arguments in __init__, should be 2 or "
                        "3 (not including self)");
    return PythonObject();
  }

  // FIXME: At this point we should check that the class we found supports all
  // the methods that we need.

  return result;
}

bool lldb_private::LLDBSWIGPythonCallThreadPlan(
    void *implementor, const char *method_name, lldb_private::Event *event,
    bool &got_error) {
  got_error = false;

  PyErr_Cleaner py_err_cleaner(false);
  PythonObject self(PyRefType::Borrowed, static_cast<PyObject *>(implementor));
  auto pfunc = self.ResolveName<PythonCallable>(method_name);

  if (!pfunc.IsAllocated())
    return false;

  PythonObject result;
  if (event != nullptr) {
    ScopedPythonObject<SBEvent> event_arg = ToSWIGWrapper(event);
    result = pfunc(event_arg.obj());
  } else
    result = pfunc();

  if (PyErr_Occurred()) {
    got_error = true;
    printf("Return value was neither false nor true for call to %s.\n",
           method_name);
    PyErr_Print();
    return false;
  }

  if (result.get() == Py_True)
    return true;
  else if (result.get() == Py_False)
    return false;

  // Somebody returned the wrong thing...
  got_error = true;
  printf("Wrong return value type for call to %s.\n", method_name);
  return false;
}

PythonObject lldb_private::LLDBSwigPythonCreateScriptedBreakpointResolver(
    const char *python_class_name, const char *session_dictionary_name,
    const StructuredDataImpl &args_impl,
    const lldb::BreakpointSP &breakpoint_sp) {

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

  PythonObject result =
      pfunc(ToSWIGWrapper(breakpoint_sp), ToSWIGWrapper(args_impl), dict);
  // FIXME: At this point we should check that the class we found supports all
  // the methods that we need.

  if (result.IsAllocated()) {
    // Check that __callback__ is defined:
    auto callback_func = result.ResolveName<PythonCallable>("__callback__");
    if (callback_func.IsAllocated())
      return result;
  }
  return PythonObject();
}

unsigned int lldb_private::LLDBSwigPythonCallBreakpointResolver(
    void *implementor, const char *method_name,
    lldb_private::SymbolContext *sym_ctx) {
  PyErr_Cleaner py_err_cleaner(false);
  PythonObject self(PyRefType::Borrowed, static_cast<PyObject *>(implementor));
  auto pfunc = self.ResolveName<PythonCallable>(method_name);

  if (!pfunc.IsAllocated())
    return 0;

  PythonObject result = sym_ctx ? pfunc(ToSWIGWrapper(*sym_ctx)) : pfunc();

  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
    return 0;
  }

  // The callback will return a bool, but we're need to also return ints
  // so we're squirrelling the bool through as an int...  And if you return
  // nothing, we'll continue.
  if (strcmp(method_name, "__callback__") == 0) {
    if (result.get() == Py_False)
      return 0;
    else
      return 1;
  }

  long long ret_val = unwrapOrSetPythonException(As<long long>(result));

  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
    return 0;
  }

  return ret_val;
}

PythonObject lldb_private::LLDBSwigPythonCreateScriptedStopHook(
    lldb::TargetSP target_sp, const char *python_class_name,
    const char *session_dictionary_name, const StructuredDataImpl &args_impl,
    Status &error) {
  if (python_class_name == NULL || python_class_name[0] == '\0') {
    error.SetErrorString("Empty class name.");
    return PythonObject();
  }
  if (!session_dictionary_name) {
    error.SetErrorString("No session dictionary");
    return PythonObject();
  }

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_class_name, dict);

  if (!pfunc.IsAllocated()) {
    error.SetErrorStringWithFormat("Could not find class: %s.",
                                   python_class_name);
    return PythonObject();
  }

  PythonObject result =
      pfunc(ToSWIGWrapper(target_sp), ToSWIGWrapper(args_impl), dict);

  if (result.IsAllocated()) {
    // Check that the handle_stop callback is defined:
    auto callback_func = result.ResolveName<PythonCallable>("handle_stop");
    if (callback_func.IsAllocated()) {
      if (auto args_info = callback_func.GetArgInfo()) {
        size_t num_args = (*args_info).max_positional_args;
        if (num_args != 2) {
          error.SetErrorStringWithFormat(
              "Wrong number of args for "
              "handle_stop callback, should be 2 (excluding self), got: %zu",
              num_args);
          return PythonObject();
        } else
          return result;
      } else {
        error.SetErrorString("Couldn't get num arguments for handle_stop "
                             "callback.");
        return PythonObject();
      }
      return result;
    } else {
      error.SetErrorStringWithFormat("Class \"%s\" is missing the required "
                                     "handle_stop callback.",
                                     python_class_name);
    }
  }
  return PythonObject();
}

bool lldb_private::LLDBSwigPythonStopHookCallHandleStop(
    void *implementor, lldb::ExecutionContextRefSP exc_ctx_sp,
    lldb::StreamSP stream) {
  // handle_stop will return a bool with the meaning "should_stop"...
  // If you return nothing we'll assume we are going to stop.
  // Also any errors should return true, since we should stop on error.

  PyErr_Cleaner py_err_cleaner(false);
  PythonObject self(PyRefType::Borrowed, static_cast<PyObject *>(implementor));
  auto pfunc = self.ResolveName<PythonCallable>("handle_stop");

  if (!pfunc.IsAllocated())
    return true;

  auto *sb_stream = new lldb::SBStream();
  PythonObject sb_stream_arg =
      ToSWIGWrapper(std::unique_ptr<lldb::SBStream>(sb_stream));
  PythonObject result =
      pfunc(ToSWIGWrapper(std::move(exc_ctx_sp)), sb_stream_arg);

  if (PyErr_Occurred()) {
    stream->PutCString("Python error occurred handling stop-hook.");
    PyErr_Print();
    PyErr_Clear();
    return true;
  }

  // Now add the result to the output stream.  SBStream only
  // makes an internally help StreamString which I can't interpose, so I
  // have to copy it over here.
  stream->PutCString(sb_stream->GetData());

  if (result.get() == Py_False)
    return false;
  else
    return true;
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

size_t lldb_private::LLDBSwigPython_CalculateNumChildren(PyObject * implementor,
                                                         uint32_t max) {
  PythonObject self(PyRefType::Borrowed, implementor);
  auto pfunc = self.ResolveName<PythonCallable>("num_children");

  if (!pfunc.IsAllocated())
    return 0;

  auto arg_info = pfunc.GetArgInfo();
  if (!arg_info) {
    llvm::consumeError(arg_info.takeError());
    return 0;
  }

  size_t ret_val;
  if (arg_info.get().max_positional_args < 1)
    ret_val = unwrapOrSetPythonException(As<long long>(pfunc.Call()));
  else
    ret_val = unwrapOrSetPythonException(
        As<long long>(pfunc.Call(PythonInteger(max))));

  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
    return 0;
  }

  if (arg_info.get().max_positional_args < 1)
    ret_val = std::min(ret_val, static_cast<size_t>(max));

  return ret_val;
}

PyObject *lldb_private::LLDBSwigPython_GetChildAtIndex(PyObject * implementor,
                                                       uint32_t idx) {
  PyErr_Cleaner py_err_cleaner(true);

  PythonObject self(PyRefType::Borrowed, implementor);
  auto pfunc = self.ResolveName<PythonCallable>("get_child_at_index");

  if (!pfunc.IsAllocated())
    return nullptr;

  PythonObject result = pfunc(PythonInteger(idx));

  if (!result.IsAllocated())
    return nullptr;

  lldb::SBValue *sbvalue_ptr = nullptr;
  if (SWIG_ConvertPtr(result.get(), (void **)&sbvalue_ptr,
                      SWIGTYPE_p_lldb__SBValue, 0) == -1)
    return nullptr;

  if (sbvalue_ptr == nullptr)
    return nullptr;

  return result.release();
}

int lldb_private::LLDBSwigPython_GetIndexOfChildWithName(
    PyObject * implementor, const char *child_name) {
  PyErr_Cleaner py_err_cleaner(true);

  PythonObject self(PyRefType::Borrowed, implementor);
  auto pfunc = self.ResolveName<PythonCallable>("get_child_index");

  if (!pfunc.IsAllocated())
    return UINT32_MAX;

  llvm::Expected<PythonObject> result = pfunc.Call(PythonString(child_name));

  long long retval =
      unwrapOrSetPythonException(As<long long>(std::move(result)));

  if (PyErr_Occurred()) {
    PyErr_Clear(); // FIXME print this? do something else
    return UINT32_MAX;
  }

  if (retval >= 0)
    return (uint32_t)retval;

  return UINT32_MAX;
}

bool lldb_private::LLDBSwigPython_UpdateSynthProviderInstance(PyObject *
                                                              implementor) {
  bool ret_val = false;

  static char callee_name[] = "update";

  PyObject *py_return =
      LLDBSwigPython_CallOptionalMember(implementor, callee_name);

  if (py_return == Py_True)
    ret_val = true;

  Py_XDECREF(py_return);

  return ret_val;
}

bool lldb_private::LLDBSwigPython_MightHaveChildrenSynthProviderInstance(
    PyObject * implementor) {
  bool ret_val = false;

  static char callee_name[] = "has_children";

  PyObject *py_return =
      LLDBSwigPython_CallOptionalMember(implementor, callee_name, Py_True);

  if (py_return == Py_True)
    ret_val = true;

  Py_XDECREF(py_return);

  return ret_val;
}

PyObject *lldb_private::LLDBSwigPython_GetValueSynthProviderInstance(
    PyObject * implementor) {
  PyObject *ret_val = nullptr;

  static char callee_name[] = "get_value";

  PyObject *py_return =
      LLDBSwigPython_CallOptionalMember(implementor, callee_name, Py_None);

  if (py_return == Py_None || py_return == nullptr)
    ret_val = nullptr;

  lldb::SBValue *sbvalue_ptr = NULL;

  if (SWIG_ConvertPtr(py_return, (void **)&sbvalue_ptr,
                      SWIGTYPE_p_lldb__SBValue, 0) == -1)
    ret_val = nullptr;
  else if (sbvalue_ptr == NULL)
    ret_val = nullptr;
  else
    ret_val = py_return;

  Py_XDECREF(py_return);
  return ret_val;
}

void *lldb_private::LLDBSWIGPython_CastPyObjectToSBData(PyObject * data) {
  lldb::SBData *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBData, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::LLDBSWIGPython_CastPyObjectToSBError(PyObject * data) {
  lldb::SBError *sb_ptr = nullptr;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBError, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::LLDBSWIGPython_CastPyObjectToSBValue(PyObject * data) {
  lldb::SBValue *sb_ptr = NULL;

  int valid_cast =
      SWIG_ConvertPtr(data, (void **)&sb_ptr, SWIGTYPE_p_lldb__SBValue, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

void *lldb_private::LLDBSWIGPython_CastPyObjectToSBMemoryRegionInfo(PyObject *
                                                                    data) {
  lldb::SBMemoryRegionInfo *sb_ptr = NULL;

  int valid_cast = SWIG_ConvertPtr(data, (void **)&sb_ptr,
                                   SWIGTYPE_p_lldb__SBMemoryRegionInfo, 0);

  if (valid_cast == -1)
    return NULL;

  return sb_ptr;
}

bool lldb_private::LLDBSwigPythonCallCommand(
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
  PythonObject debugger_arg = ToSWIGWrapper(std::move(debugger));
  auto cmd_retobj_arg = ToSWIGWrapper(cmd_retobj);

  if (argc.get().max_positional_args < 5u)
    pfunc(debugger_arg, PythonString(args), cmd_retobj_arg.obj(), dict);
  else
    pfunc(debugger_arg, PythonString(args),
          ToSWIGWrapper(std::move(exe_ctx_ref_sp)), cmd_retobj_arg.obj(), dict);

  return true;
}

bool lldb_private::LLDBSwigPythonCallCommandObject(
    PyObject *implementor, lldb::DebuggerSP debugger, const char *args,
    lldb_private::CommandReturnObject &cmd_retobj,
    lldb::ExecutionContextRefSP exe_ctx_ref_sp) {

  PyErr_Cleaner py_err_cleaner(true);

  PythonObject self(PyRefType::Borrowed, implementor);
  auto pfunc = self.ResolveName<PythonCallable>("__call__");

  if (!pfunc.IsAllocated())
    return false;

  auto cmd_retobj_arg = ToSWIGWrapper(cmd_retobj);

  pfunc(ToSWIGWrapper(std::move(debugger)), PythonString(args),
        ToSWIGWrapper(exe_ctx_ref_sp), cmd_retobj_arg.obj());

  return true;
}

PythonObject lldb_private::LLDBSWIGPythonCreateOSPlugin(
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

  return pfunc(ToSWIGWrapper(process_sp));
}

PythonObject lldb_private::LLDBSWIGPython_CreateFrameRecognizer(
    const char *python_class_name, const char *session_dictionary_name) {
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

  return pfunc();
}

PyObject *lldb_private::LLDBSwigPython_GetRecognizedArguments(
    PyObject * implementor, const lldb::StackFrameSP &frame_sp) {
  static char callee_name[] = "get_recognized_arguments";

  PythonObject arg = ToSWIGWrapper(frame_sp);

  PythonString str(callee_name);
  PyObject *result =
      PyObject_CallMethodObjArgs(implementor, str.get(), arg.get(), NULL);
  return result;
}

void *lldb_private::LLDBSWIGPython_GetDynamicSetting(
    void *module, const char *setting, const lldb::TargetSP &target_sp) {
  if (!module || !setting)
    Py_RETURN_NONE;

  PyErr_Cleaner py_err_cleaner(true);
  PythonObject py_module(PyRefType::Borrowed, (PyObject *)module);
  auto pfunc = py_module.ResolveName<PythonCallable>("get_dynamic_setting");

  if (!pfunc.IsAllocated())
    Py_RETURN_NONE;

  auto result = pfunc(ToSWIGWrapper(target_sp), PythonString(setting));

  return result.release();
}

bool lldb_private::LLDBSWIGPythonRunScriptKeywordProcess(
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

  auto result = pfunc(ToSWIGWrapper(process), dict);

  output = result.Str().GetString().str();

  return true;
}

llvm::Optional<std::string> lldb_private::LLDBSWIGPythonRunScriptKeywordThread(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ThreadSP thread) {
  if (python_function_name == NULL || python_function_name[0] == '\0' ||
      !session_dictionary_name)
    return llvm::None;

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return llvm::None;

  auto result = pfunc(ToSWIGWrapper(std::move(thread)), dict);

  return result.Str().GetString().str();
}

bool lldb_private::LLDBSWIGPythonRunScriptKeywordTarget(
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

  auto result = pfunc(ToSWIGWrapper(target), dict);

  output = result.Str().GetString().str();

  return true;
}

llvm::Optional<std::string> lldb_private::LLDBSWIGPythonRunScriptKeywordFrame(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::StackFrameSP frame) {
  if (python_function_name == NULL || python_function_name[0] == '\0' ||
      !session_dictionary_name)
    return llvm::None;

  PyErr_Cleaner py_err_cleaner(true);

  auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(
      session_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      python_function_name, dict);

  if (!pfunc.IsAllocated())
    return llvm::None;

  auto result = pfunc(ToSWIGWrapper(std::move(frame)), dict);

  return result.Str().GetString().str();
}

bool lldb_private::LLDBSWIGPythonRunScriptKeywordValue(
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

  auto result = pfunc(ToSWIGWrapper(value), dict);

  output = result.Str().GetString().str();

  return true;
}

bool lldb_private::LLDBSwigPythonCallModuleInit(
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

  pfunc(ToSWIGWrapper(std::move(debugger)), dict);

  return true;
}

lldb::ValueObjectSP lldb_private::LLDBSWIGPython_GetValueObjectSPFromSBValue(
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
%}
