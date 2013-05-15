//===---------------------PythonPointer.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_PythonPointer_h_
#define utility_PythonPointer_h_

#include <algorithm>

#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif

namespace lldb_private {

template<class T>
class PythonPointer
{
public: 
    typedef PyObject* element_type; 
private:
    element_type*      ptr_;
    bool my_ref;
public:
    
    PythonPointer(element_type p, bool steal_ref = false) :
    ptr_(p),
    my_ref(!steal_ref)
    {
        if (my_ref)
            Py_INCREF(ptr_);
    }
    
    PythonPointer(const PythonPointer& r, bool steal_ref = false) :
    ptr_(r.ptr_),
    my_ref(!steal_ref)
    {
        if (my_ref)
            Py_INCREF(ptr_);
    }

    ~PythonPointer()
    {
        if (my_ref)
            Py_XDECREF(ptr_);
    }
    
    PythonPointer
    StealReference()
    {
        return PythonPointer(ptr_,true);
    }
    
    PythonPointer
    DuplicateReference()
    {
        return PythonPointer(ptr_, false);
    }

    element_type get() const {return ptr_;}
    
    bool IsNull() { return ptr_ == NULL; }
    bool IsNone() { return ptr_ == Py_None; }
    
    operator PyObject* () { return ptr_; }
};

} // namespace lldb

#endif  // utility_PythonPointer_h_
