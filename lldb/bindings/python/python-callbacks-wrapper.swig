/*
  Wrapper that holds static functions used for callback typemaps in LLDB.
*/

%header %{
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
