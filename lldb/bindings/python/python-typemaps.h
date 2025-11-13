#ifndef LLDB_BINDINGS_PYTHON_PYTHON_TYPEMAPS_H
#define LLDB_BINDINGS_PYTHON_PYTHON_TYPEMAPS_H

#include <Python.h>

#if SWIG_VERSION < 0x040100
// Defined here instead of a .swig file because SWIG 2 doesn't support
// explicit deleted functions.
struct Py_buffer_RAII {
  Py_buffer buffer = {};
  Py_buffer_RAII(){};
  Py_buffer &operator=(const Py_buffer_RAII &) = delete;
  Py_buffer_RAII(const Py_buffer_RAII &) = delete;
  ~Py_buffer_RAII() {
    if (buffer.obj)
      PyBuffer_Release(&buffer);
  }
};
#endif

#endif // LLDB_BINDINGS_PYTHON_PYTHON_TYPEMAPS_H
