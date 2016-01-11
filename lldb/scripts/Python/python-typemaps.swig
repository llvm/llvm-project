/* Typemap definitions, to allow SWIG to properly handle 'char**' data types. */

%typemap(in) char ** {
  /* Check if is a list  */
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $1 = (char **) malloc((size+1) * sizeof(char*));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyString_Check(o))
        $1[i] = PyString_AsString(o);
      else {
        PyErr_SetString(PyExc_TypeError,"list must contain strings");
        free($1);
        return NULL;
      }
    }
    $1[i] = 0;
  } else if ($input == Py_None) {
    $1 =  NULL;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

%typemap(in) lldb::tid_t {
  using namespace lldb_private;
  if (PythonInteger::Check($input))
  {
    PythonInteger py_int(PyRefType::Borrowed, $input);
    $1 = static_cast<lldb::tid_t>(py_int.GetInteger());
  }
  else
  {
    PyErr_SetString(PyExc_ValueError, "Expecting an integer");
    return nullptr;
  }
}

%typemap(typecheck) char ** {
  /* Check if is a list  */
  $1 = 1;
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (!PyString_Check(o)) { $1 = 0; }
    }
  }
  else
  {
    $1 = ( ($input == Py_None) ? 1 : 0);
  }
}

%typemap(freearg) char** {
  free((char *) $1);
}

%typemap(out) char** {
  int len;
  int i;
  len = 0;
  while ($1[len]) len++;
  using namespace lldb_private;
  PythonList list(len);
  for (i = 0; i < len; i++)
    list.SetItemAtIndex(i, PythonString($1[i]));
  $result = list.release();
}

%typemap(in) char const ** {
  /* Check if is a list  */
  using namespace lldb_private;
  if (PythonList::Check($input)) {
    PythonList py_list(PyRefType::Borrowed, $input);
    int size = py_list.GetSize();

    $1 = (char**)malloc((size+1)*sizeof(char*));
    for (int i = 0; i < size; i++) {
      PythonObject o = py_list.GetItemAtIndex(i);
      if (!PythonString::Check(o.get())) {
        PyErr_SetString(PyExc_TypeError,"list must contain strings");
        free($1);
        return nullptr;
      }
      auto py_str = o.AsType<PythonString>();
      $1[i] = const_cast<char*>(py_str.GetString().data());
    }

    $1[size] = 0;
  } else if ($input == Py_None) {
    $1 = nullptr;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return nullptr;
  }
}

%typemap(typecheck) char const ** {
  /* Check if is a list  */
  $1 = 1;
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (!PyString_Check(o)) { $1 = 0; }
    }
  }
  else
  {
    $1 = ( ($input == Py_None) ? 1 : 0);
  }
}

%typemap(freearg) char const ** {
  free((char *) $1);
}

%typemap(out) char const ** {
  int len;
  int i;
  len = 0;
  while ($1[len]) len++;
  $result = PyList_New(len);
  for (i = 0; i < len; i++) {
    PyList_SetItem($result, i, PyString_FromString($1[i]));
  }
}

/* Typemap definitions to allow SWIG to properly handle char buffer. */

// typemap for a char buffer
// See also SBThread::GetStopDescription.
%typemap(in) (char *dst, size_t dst_len) {
   if (!PyInt_Check($input)) {
       PyErr_SetString(PyExc_ValueError, "Expecting an integer");
       return NULL;
   }
   $2 = PyInt_AsLong($input);
   if ($2 <= 0) {
       PyErr_SetString(PyExc_ValueError, "Positive integer expected");
       return NULL;
   }
   $1 = (char *) malloc($2);
}

// Return the char buffer.  Discarding any previous return result
// See also SBThread::GetStopDescription.
%typemap(argout) (char *dst, size_t dst_len) {
   Py_XDECREF($result);   /* Blow away any previous result */
   if (result == 0) {
      $result = Py_None;
      Py_INCREF($result);
   } else {
      llvm::StringRef ref(static_cast<const char*>($1), result);
      lldb_private::PythonString string(ref);
      $result = string.release();
   }
   free($1);
}


// typemap for an outgoing buffer
// See also SBEvent::SBEvent(uint32_t event, const char *cstr, uint32_t cstr_len).
%typemap(in) (const char *cstr, uint32_t cstr_len) {
   if (PyString_Check($input)) {
      $1 = (char *) PyString_AsString($input);
      $2 = PyString_Size($input);
   }
   else if(PyByteArray_Check($input)) {
      $1 = (char *) PyByteArray_AsString($input);
      $2 = PyByteArray_Size($input);
   }
   else {
      PyErr_SetString(PyExc_ValueError, "Expecting a string");
      return NULL;
   }
}
// Ditto for SBProcess::PutSTDIN(const char *src, size_t src_len).
%typemap(in) (const char *src, size_t src_len) {
   if (PyString_Check($input)) {
      $1 = (char *) PyString_AsString($input);
      $2 = PyString_Size($input);
   }
   else if(PyByteArray_Check($input)) {
      $1 = (char *) PyByteArray_AsString($input);
      $2 = PyByteArray_Size($input);
   }
   else {
      PyErr_SetString(PyExc_ValueError, "Expecting a string");
      return NULL;
   }
}
// And SBProcess::WriteMemory.
%typemap(in) (const void *buf, size_t size) {
   if (PyString_Check($input)) {
      $1 = (void *) PyString_AsString($input);
      $2 = PyString_Size($input);
   }
   else if(PyByteArray_Check($input)) {
      $1 = (void *) PyByteArray_AsString($input);
      $2 = PyByteArray_Size($input);
   }
   else {
      PyErr_SetString(PyExc_ValueError, "Expecting a string");
      return NULL;
   }
}

// For SBDebugger::DispatchInput
%typemap(in) (const void *data, size_t data_len) {
   if (PyString_Check($input)) {
      $1 = static_cast<void *>(PyString_AsString($input));
      $2 = PyString_Size($input);
   }
   else if(PyByteArray_Check($input)) {
      $1 = static_cast<void *>(PyByteArray_AsString($input));
      $2 = PyByteArray_Size($input);
   }
   else {
      PyErr_SetString(PyExc_ValueError, "Expecting a string or byte array");
      return NULL;
   }
}

// typemap for an incoming buffer
// See also SBProcess::ReadMemory.
%typemap(in) (void *buf, size_t size) {
   if (PyInt_Check($input)) {
      $2 = PyInt_AsLong($input);
   } else if (PyLong_Check($input)) {
      $2 = PyLong_AsLong($input);
   } else {
      PyErr_SetString(PyExc_ValueError, "Expecting an integer or long object");
      return NULL;
   }
   if ($2 <= 0) {
       PyErr_SetString(PyExc_ValueError, "Positive integer expected");
       return NULL;
   }
   $1 = (void *) malloc($2);
}

// Return the buffer.  Discarding any previous return result
// See also SBProcess::ReadMemory.
%typemap(argout) (void *buf, size_t size) {
   Py_XDECREF($result);   /* Blow away any previous result */
   if (result == 0) {
      $result = Py_None;
      Py_INCREF($result);
   } else {
      llvm::StringRef ref(static_cast<const char*>($1), result);
      lldb_private::PythonString string(ref);
      $result = string.release();
   }
   free($1);
}

// these typemaps allow Python users to pass list objects
// and have them turn into C++ arrays (this is useful, for instance
// when creating SBData objects from lists of numbers)
%typemap(in) (uint64_t* array, size_t array_len) {
  /* Check if is a list  */
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $2 = size;
    $1 = (uint64_t*) malloc(size * sizeof(uint64_t));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyInt_Check(o)) {
        $1[i] = PyInt_AsLong(o);
      }
      else if (PyLong_Check(o)) {
        $1[i] = PyLong_AsUnsignedLongLong(o);
      }
      else {
        PyErr_SetString(PyExc_TypeError,"list must contain numbers");
        free($1);
        return NULL;
      }

      if (PyErr_Occurred()) {
        free($1);
        return NULL;
      }
    }
  } else if ($input == Py_None) {
    $1 =  NULL;
    $2 = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

%typemap(freearg) (uint64_t* array, size_t array_len) {
  free($1);
}

%typemap(in) (uint32_t* array, size_t array_len) {
  /* Check if is a list  */
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $2 = size;
    $1 = (uint32_t*) malloc(size * sizeof(uint32_t));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyInt_Check(o)) {
        $1[i] = PyInt_AsLong(o);
      }
      else if (PyLong_Check(o)) {
        $1[i] = PyLong_AsUnsignedLong(o);
      }
      else {
        PyErr_SetString(PyExc_TypeError,"list must contain numbers");
        free($1);
        return NULL;
      }

      if (PyErr_Occurred()) {
        free($1);
        return NULL;
      }
    }
  } else if ($input == Py_None) {
    $1 =  NULL;
    $2 = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

%typemap(freearg) (uint32_t* array, size_t array_len) {
  free($1);
}

%typemap(in) (int64_t* array, size_t array_len) {
  /* Check if is a list  */
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $2 = size;
    $1 = (int64_t*) malloc(size * sizeof(int64_t));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyInt_Check(o)) {
        $1[i] = PyInt_AsLong(o);
      }
      else if (PyLong_Check(o)) {
        $1[i] = PyLong_AsLongLong(o);
      }
      else {
        PyErr_SetString(PyExc_TypeError,"list must contain numbers");
        free($1);
        return NULL;
      }

      if (PyErr_Occurred()) {
        free($1);
        return NULL;
      }
    }
  } else if ($input == Py_None) {
    $1 =  NULL;
    $2 = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

%typemap(freearg) (int64_t* array, size_t array_len) {
  free($1);
}

%typemap(in) (int32_t* array, size_t array_len) {
  /* Check if is a list  */
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $2 = size;
    $1 = (int32_t*) malloc(size * sizeof(int32_t));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyInt_Check(o)) {
        $1[i] = PyInt_AsLong(o);
      }
      else if (PyLong_Check(o)) {
        $1[i] = PyLong_AsLong(o);
      }
      else {
        PyErr_SetString(PyExc_TypeError,"list must contain numbers");
        free($1);
        return NULL;
      }

      if (PyErr_Occurred()) {
        free($1);
        return NULL;
      }
    }
  } else if ($input == Py_None) {
    $1 =  NULL;
    $2 = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

%typemap(freearg) (int32_t* array, size_t array_len) {
  free($1);
}

%typemap(in) (double* array, size_t array_len) {
  /* Check if is a list  */
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $2 = size;
    $1 = (double*) malloc(size * sizeof(double));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyFloat_Check(o)) {
        $1[i] = PyFloat_AsDouble(o);
      }
      else {
        PyErr_SetString(PyExc_TypeError,"list must contain floating-point numbers");
        free($1);
        return NULL;
      }
    }
  } else if ($input == Py_None) {
    $1 =  NULL;
    $2 = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

%typemap(freearg) (double* array, size_t array_len) {
  free($1);
}

// these typemaps wrap SBModule::GetVersion() from requiring a memory buffer
// to the more Pythonic style where a list is returned and no previous allocation
// is necessary - this will break if more than 50 versions are ever returned
%typemap(typecheck) (uint32_t *versions, uint32_t num_versions) {
    $1 = ($input == Py_None ? 1 : 0);
}

%typemap(in, numinputs=0) (uint32_t *versions) {
    $1 = (uint32_t*)malloc(sizeof(uint32_t) * 50);
}

%typemap(in, numinputs=0) (uint32_t num_versions) {
    $1 = 50;
}

%typemap(argout) (uint32_t *versions, uint32_t num_versions) {
    uint32_t count = result;
    if (count >= $2)
        count = $2;
    PyObject* list = PyList_New(count);
    for (uint32_t j = 0; j < count; j++)
    {
        if ($1[j] < UINT32_MAX)
        {
            PyObject* item = PyInt_FromLong($1[j]);
            int ok = PyList_SetItem(list,j,item);
            if (ok != 0)
            {
                $result = Py_None;
                break;
            }
        }
        else
            break;
    }
    $result = list;
}

%typemap(freearg) (uint32_t *versions) {
    free($1);
}


// For Log::LogOutputCallback
%typemap(in) (lldb::LogOutputCallback log_callback, void *baton) {
  if (!($input == Py_None || PyCallable_Check(reinterpret_cast<PyObject*>($input)))) {
    PyErr_SetString(PyExc_TypeError, "Need a callable object or None!");
    return NULL;
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
  $1 = $1 || PyCallable_Check(reinterpret_cast<PyObject*>($input));
}

%typemap(in) FILE * {
   using namespace lldb_private;
   if ($input == Py_None)
      $1 = nullptr;
   else if (!lldb_private::PythonFile::Check($input)) {
      int fd = PyObject_AsFileDescriptor($input);
      PythonObject py_input(PyRefType::Borrowed, $input);
      PythonString py_mode = py_input.GetAttributeValue("mode").AsType<PythonString>();

      if (-1 != fd && py_mode.IsValid()) {
         FILE *f;
         if ((f = fdopen(fd, py_mode.GetString().str().c_str())))
            $1 = f;
         else
            PyErr_SetString(PyExc_TypeError, strerror(errno));
      } else {
         PyErr_SetString(PyExc_TypeError,"not a file-like object");
         return nullptr;
      }
   }
   else
   {
      PythonFile py_file(PyRefType::Borrowed, $input);
      File file;
      if (!py_file.GetUnderlyingFile(file))
         return nullptr;

      $1 = file.GetStream();
   }
}

%typemap(out) FILE * {
   char mode[4] = {0};
#ifdef __MACOSX__
   int i = 0;
   short flags = $1->_flags;

   if (flags & __SRD)
      mode[i++] = 'r';
   else if (flags & __SWR)
      mode[i++] = 'w';
   else // if (flags & __SRW)
      mode[i++] = 'a';
#endif
   using namespace lldb_private;
   File file($1, false);
   PythonFile py_file(file, mode);
   $result = py_file.release();
   if (!$result)
   {
       $result = Py_None;
       Py_INCREF(Py_None);
   }
}

%typemap(in) (const char* string, int len) {
    using namespace lldb_private;
    if ($input == Py_None)
    {
        $1 = NULL;
        $2 = 0;
    }
    else if (PythonString::Check($input))
    {
        PythonString py_str(PyRefType::Borrowed, $input);
        llvm::StringRef str = py_str.GetString();
        $1 = const_cast<char*>(str.data());
        $2 = str.size();
        // In Python 2, if $input is a PyUnicode object then this
        // will trigger a Unicode -> String conversion, in which
        // case the `PythonString` will now own the PyString.  Thus
        // if it goes out of scope, the data will be deleted.  The
        // only way to avoid this is to leak the Python object in
        // that case.  Note that if there was no conversion, then
        // releasing the string will not leak anything, since we
        // created this as a borrowed reference.
        py_str.release();
    }
    else
    {
        PyErr_SetString(PyExc_TypeError,"not a string-like object");
        return NULL;
    }
}
