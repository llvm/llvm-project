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
    
    class PythonObject
    {
    public:
        PythonObject () :
            m_py_obj(NULL)
        {
        }
        
        PythonObject (PyObject* py_obj) :
            m_py_obj(NULL)
        {
            Reset (py_obj);
        }
        
        PythonObject (const PythonObject &rhs) :
            m_py_obj(NULL)
        {
            Reset (rhs.m_py_obj);
        }
        
        PythonObject (const lldb::ScriptInterpreterObjectSP &script_object_sp);

        virtual
        ~PythonObject ()
        {
            Reset (NULL);
        }

        const PythonObject &
        operator = (const PythonObject &rhs)
        {
            if (this != &rhs)
                Reset (rhs.m_py_obj);
            return *this;
        }

        bool
        Reset (const PythonObject &object)
        {
            return Reset(object.GetPythonObject());
        }

        virtual bool
        Reset (PyObject* py_obj = NULL)
        {
            if (py_obj != m_py_obj)
            {
                Py_XDECREF(m_py_obj);
                m_py_obj = py_obj;
                Py_XINCREF(m_py_obj);
            }
            return true;
        }
        
        void
        Dump () const
        {
            if (m_py_obj)
                _PyObject_Dump (m_py_obj);
            else
                puts ("NULL");
        }
        
        void
        Dump (Stream &strm) const;

        PyObject*
        GetPythonObject () const
        {
            return m_py_obj;
        }
        
        operator bool () const
        {
            return m_py_obj != NULL;
        }
        
    protected:
        PyObject* m_py_obj;
    };
    
    class PythonString: public PythonObject
    {
    public:
        
        PythonString ();
        PythonString (PyObject *o);
        PythonString (const PythonObject &object);
        PythonString (const lldb::ScriptInterpreterObjectSP &script_object_sp);
        PythonString (const char* string);
        virtual ~PythonString ();
        
        virtual bool
        Reset (PyObject* py_obj = NULL);

        const char*
        GetString() const;

        size_t
        GetSize() const;

        void
        SetString (const char* string);        
    };
    
    class PythonInteger: public PythonObject
    {
    public:
        
        PythonInteger ();
        PythonInteger (PyObject* py_obj);
        PythonInteger (const PythonObject &object);
        PythonInteger (const lldb::ScriptInterpreterObjectSP &script_object_sp);
        PythonInteger (int64_t value);
        virtual ~PythonInteger ();
        
        virtual bool
        Reset (PyObject* py_obj = NULL);
        
        int64_t
        GetInteger();
        
        void
        SetInteger (int64_t value);
    };
    
    class PythonList: public PythonObject
    {
    public:
        
        PythonList ();
        PythonList (PyObject* py_obj);
        PythonList (const PythonObject &object);
        PythonList (const lldb::ScriptInterpreterObjectSP &script_object_sp);
        PythonList (uint32_t count);
        virtual ~PythonList ();
        
        virtual bool
        Reset (PyObject* py_obj = NULL);
        
        uint32_t
        GetSize();
        
        PythonObject
        GetItemAtIndex (uint32_t index);
        
        void
        SetItemAtIndex (uint32_t index, const PythonObject &object);
        
        void
        AppendItem (const PythonObject &object);
    };
    
    class PythonDictionary: public PythonObject
    {
    public:
        
        PythonDictionary ();
        PythonDictionary (PyObject* object);
        PythonDictionary (const PythonObject &object);
        PythonDictionary (const lldb::ScriptInterpreterObjectSP &script_object_sp);
        virtual ~PythonDictionary ();
        
        virtual bool
        Reset (PyObject* object = NULL);
        
        uint32_t GetSize();
        
        PythonObject
        GetItemForKey (const PythonString &key) const;
        
        const char *
        GetItemForKeyAsString (const PythonString &key, const char *fail_value = NULL) const;

        int64_t
        GetItemForKeyAsInteger (const PythonString &key, int64_t fail_value = 0) const;

        PythonObject
        GetItemForKey (const char *key) const;

        typedef bool (*DictionaryIteratorCallback)(PythonString* key, PythonDictionary* dict);
        
        PythonList
        GetKeys () const;
        
        PythonString
        GetKeyAtPosition (uint32_t pos) const;
        
        PythonObject
        GetValueAtPosition (uint32_t pos) const;
        
        void
        SetItemForKey (const PythonString &key, const PythonObject& value);
    };
    
} // namespace lldb_private

#endif  // liblldb_PythonDataObjects_h_
