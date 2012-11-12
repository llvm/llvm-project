//===-- PythonDataObjects.h----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PythonDataObjects_h_
#define liblldb_PythonDataObjects_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-defines.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Flags.h"
#include "lldb/Interpreter/OptionValue.h"
#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif

namespace lldb_private {
    
    class PythonRefCountedObject
    {
    public:
        PythonRefCountedObject (PyObject* obj = NULL) : m_object(obj)
        {
            Py_XINCREF(m_object);
        }
        
        PythonRefCountedObject (const PythonRefCountedObject &rhs) :
            m_object(rhs.m_object)
        {
            Py_XINCREF(m_object);
        }
        
        ~PythonRefCountedObject ()
        {
            Py_XDECREF(m_object);
        }

        const PythonRefCountedObject &
        operator = (const PythonRefCountedObject &rhs)
        {
            if (this != &rhs)
                Reset (rhs.m_object);
            return *this;
        }

        void
        Reset (PyObject* object = NULL)
        {
            if (object != m_object)
            {
                Py_XDECREF(m_object);
                m_object = object;
                Py_XINCREF(m_object);
            }
        }
        
        PyObject*
        GetPyhonObject () const
        {
            return m_object;
        }
        
        operator bool () const
        {
            return m_object != NULL;
        }
        
    private:
        PyObject* m_object;
    };
    
    class PythonDataString
    {
    public:
        
        PythonDataString (bool create_empty);
        PythonDataString (PyObject* object = NULL);
        PythonDataString (const char* string);
        ~PythonDataString ();
        
        const char*
        GetString() const;

        size_t
        GetSize() const;

        void
        SetString (const char* string);
        
        operator bool () const
        {
            return m_object.operator bool();
        }
        
        PyObject*
        GetPythonObject() const
        {
            return m_object.GetPyhonObject();
        }
    private:
        PythonRefCountedObject m_object;
    };
    
    class PythonDataInteger
    {
    public:
        
        PythonDataInteger (bool create_empty = true);
        PythonDataInteger (PyObject* object);
        PythonDataInteger (int64_t value);
        ~PythonDataInteger ();
        
        int64_t
        GetInteger();
        
        void
        SetInteger (int64_t value);
        
        operator bool () const
        {
            return m_object.operator bool();
        }
        
        PyObject*
        GetPythonObject() const
        {
            return m_object.GetPyhonObject();
        }
    private:
        PythonRefCountedObject m_object;
    };
    
    class PythonDataArray
    {
    public:
        
        PythonDataArray (bool create_empty = true);
        PythonDataArray (PyObject* object);
        PythonDataArray (uint32_t count);
        ~PythonDataArray ();
        
        uint32_t
        GetSize();
        
        PythonDataObject
        GetItemAtIndex (uint32_t index);
        
        void
        SetItemAtIndex (uint32_t index, const PythonDataObject &object);
        
        void
        AppendItem (const PythonDataObject &object);
        
        operator bool () const
        {
            return m_object.operator bool();
        }
        
        PyObject*
        GetPythonObject() const
        {
            return m_object.GetPyhonObject();
        }
    private:
        PythonRefCountedObject m_object;
    };
    
    class PythonDataDictionary
    {
    public:
        
        PythonDataDictionary (bool create_empty = true);
        PythonDataDictionary (PyObject* object);
        ~PythonDataDictionary ();
        
        uint32_t GetSize();
        
        PythonDataObject
        GetItemForKey (const PythonDataString &key) const;
        
        const char *
        GetItemForKeyAsString (const PythonDataString &key, const char *fail_value = NULL) const;

        int64_t
        GetItemForKeyAsInteger (const PythonDataString &key, int64_t fail_value = 0) const;

        PythonDataObject
        GetItemForKey (const char *key) const;

        typedef bool (*DictionaryIteratorCallback)(PythonDataString* key, PythonDataDictionary* dict);
        
        PythonDataArray
        GetKeys () const;
        
        PythonDataString
        GetKeyAtPosition (uint32_t pos) const;
        
        PythonDataObject
        GetValueAtPosition (uint32_t pos) const;
        
        void
        SetItemForKey (const PythonDataString &key, const PythonDataObject& value);
        
        operator bool () const
        {
            return m_object.operator bool();
        }
        
        PyObject*
        GetPythonObject() const
        {
            return m_object.GetPyhonObject();
        }
    private:
        PythonRefCountedObject m_object;
    };

    class PythonDataObject
    {
    public:
        
        PythonDataObject ();
        PythonDataObject (PyObject* object);
        
        ~PythonDataObject ();
        
        PythonDataString
        GetStringObject ();
        
        PythonDataInteger
        GetIntegerObject ();
        
        PythonDataArray
        GetArrayObject();
        
        PythonDataDictionary
        GetDictionaryObject();
        
        operator bool () const
        {
            return m_object.operator bool();
        }
        
        PyObject*
        GetPythonObject() const
        {
            return m_object.GetPyhonObject();
        }
        
    private:
        PythonRefCountedObject m_object;
    };
    
} // namespace lldb_private

#endif  // liblldb_PythonDataObjects_h_
