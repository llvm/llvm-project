/*
 Typemaps specific to callback functions in LLDB. If editing this file
 use the Python C API to access Python objects instead of using PythonDataObjects.
*/

// For Log::LogOutputCallback
%typemap(in) (lldb::LogOutputCallback log_callback, void *baton) {
  if (!($input == Py_None ||
        PyCallable_Check(reinterpret_cast<PyObject *>($input)))) {
    PyErr_SetString(PyExc_TypeError, "Need a callable object or None!");
    SWIG_fail;
  }

  // FIXME (filcab): We can't currently check if our callback is already
  // LLDBSwigPythonCallPythonLogOutputCallback (to DECREF the previous
  // baton) nor can we just remove all traces of a callback, if we want to
  // revert to a file logging mechanism.

  // Don't lose the callback reference
  Py_INCREF($input);
  $1 = LLDBSwigPythonCallPythonLogOutputCallback;
  $2 = $input;
}

%typemap(typecheck) (lldb::LogOutputCallback log_callback, void *baton) {
  $1 = $input == Py_None;
  $1 = $1 || PyCallable_Check(reinterpret_cast<PyObject *>($input));
}

// For lldb::SBDebuggerDestroyCallback
%typemap(in) (lldb::SBDebuggerDestroyCallback destroy_callback, void *baton) {
  if (!($input == Py_None ||
        PyCallable_Check(reinterpret_cast<PyObject *>($input)))) {
    PyErr_SetString(PyExc_TypeError, "Need a callable object or None!");
    SWIG_fail;
  }

  // FIXME (filcab): We can't currently check if our callback is already
  // LLDBSwigPythonCallPythonSBDebuggerTerminateCallback (to DECREF the previous
  // baton) nor can we just remove all traces of a callback, if we want to
  // revert to a file logging mechanism.

  // Don't lose the callback reference
  Py_INCREF($input);
  $1 = LLDBSwigPythonCallPythonSBDebuggerTerminateCallback;
  $2 = $input;
}

%typemap(typecheck) (lldb::SBDebuggerDestroyCallback destroy_callback, void *baton) {
  $1 = $input == Py_None;
  $1 = $1 || PyCallable_Check(reinterpret_cast<PyObject *>($input));
}

%typemap(in) (lldb::CommandOverrideCallback callback, void *baton) {
  if (!($input == Py_None ||
        PyCallable_Check(reinterpret_cast<PyObject *>($input)))) {
    PyErr_SetString(PyExc_TypeError, "Need a callable object or None!");
    SWIG_fail;
  }

  // Don't lose the callback reference
  Py_INCREF($input);
  $1 = LLDBSwigPythonCallPythonSBCommandInterpreterSetCommandOverrideCallback;
  $2 = $input;
}
%typemap(typecheck) (lldb::CommandOverrideCallback callback, void *baton) {
  $1 = $input == Py_None;
  $1 = $1 || PyCallable_Check(reinterpret_cast<PyObject *>($input));
}
// For lldb::SBPlatformLocateModuleCallback
%typemap(in) (lldb::SBPlatformLocateModuleCallback callback,
              void *callback_baton) {
  if (!($input == Py_None ||
        PyCallable_Check(reinterpret_cast<PyObject *>($input)))) {
    PyErr_SetString(PyExc_TypeError, "Need a callable object or None!");
    SWIG_fail;
  }

  if ($input == Py_None) {
    $1 = nullptr;
    $2 = nullptr;
  } else {
    PythonCallable callable = Retain<PythonCallable>($input);
    if (!callable.IsValid()) {
      PyErr_SetString(PyExc_TypeError, "Need a valid callable object");
      SWIG_fail;
    }

    llvm::Expected<PythonCallable::ArgInfo> arg_info = callable.GetArgInfo();
    if (!arg_info) {
      PyErr_SetString(PyExc_TypeError,
                      ("Could not get arguments: " +
                          llvm::toString(arg_info.takeError())).c_str());
      SWIG_fail;
    }

    if (arg_info.get().max_positional_args != 3) {
      PyErr_SetString(PyExc_TypeError, "Expected 3 argument callable object");
      SWIG_fail;
    }

    // NOTE: When this is called multiple times, this will leak the Python
    // callable object as other callbacks, because this does not call Py_DECREF
    // the object. But it should be almost zero impact since this method is
    // expected to be called only once.

    // Don't lose the callback reference
    Py_INCREF($input);

    $1 = LLDBSwigPythonCallLocateModuleCallback;
    $2 = $input;
  }
}

%typemap(typecheck) (lldb::SBPlatformLocateModuleCallback callback,
                     void *callback_baton) {
  $1 = $input == Py_None;
  $1 = $1 || PyCallable_Check(reinterpret_cast<PyObject *>($input));
}
