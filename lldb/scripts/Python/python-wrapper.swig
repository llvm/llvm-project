%wrapper %{

class PyErr_Cleaner
{
public:
    PyErr_Cleaner(bool print=false) :
        m_print(print)
    {
    }

    ~PyErr_Cleaner()
    {
        if (PyErr_Occurred())
        {
            if(m_print)
                PyErr_Print();
            PyErr_Clear();
        }
    }

private:
    bool m_print;
};

// resolve a dotted Python name in the form
// foo.bar.baz.Foobar to an actual Python object
// if pmodule is NULL, the __main__ module will be used
// as the starting point for the search

static PyObject*
ResolvePythonName(const char* name,
                  PyObject* pmodule = NULL)
{
    if (!name)
        return pmodule;

    PyErr_Cleaner pyerr_cleanup(true);  // show Python errors

    PyObject* main_dict;

    if (!pmodule)
    {
        pmodule = PyImport_AddModule ("__main__");
        if (!pmodule)
            return NULL;
    }

    if (!PyDict_Check(pmodule))
    {
        main_dict = PyModule_GetDict (pmodule);
        if (!main_dict)
            return NULL;
    }
    else
        main_dict = pmodule;

    const char* dot_pos = ::strchr(name, '.');

    PyObject *dest_object;
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    if (!dot_pos)
    {
        dest_object = NULL;   
        while (PyDict_Next (main_dict, &pos, &key, &value))
        {
            // We have stolen references to the key and value objects in the dictionary; we need to increment 
            // them now so that Python's garbage collector doesn't collect them out from under us.
            Py_INCREF (key);
            Py_INCREF (value);
            if (strcmp (PyString_AsString (key), name) == 0)
            {
                dest_object = value;
                break;
            }
        }        
        if (!dest_object || dest_object == Py_None)
            return NULL;
        return dest_object;
    }
    else
    {
        size_t len = dot_pos - name;
        std::string piece(name,len);
        pmodule = ResolvePythonName(piece.c_str(), main_dict);
        if (!pmodule)
            return NULL;
        name = dot_pos+1;
        return ResolvePythonName(dot_pos+1,pmodule); // tail recursion.. should be optimized by the compiler
    }
}

static PyObject*
FindSessionDictionary(const char *session_dictionary_name)
{
    return ResolvePythonName(session_dictionary_name, NULL);
}

// This function is called by lldb_private::ScriptInterpreterPython::BreakpointCallbackFunction(...)
// and is used when a script command is attached to a breakpoint for execution.

SWIGEXPORT bool
LLDBSwigPythonBreakpointCallbackFunction 
(
    const char *python_function_name,
    const char *session_dictionary_name,
    const lldb::StackFrameSP& frame_sp, 
    const lldb::BreakpointLocationSP& bp_loc_sp
)
{
    lldb::SBFrame sb_frame (frame_sp);
    lldb::SBBreakpointLocation sb_bp_loc(bp_loc_sp);

    bool stop_at_breakpoint = true;
    PyObject *Frame_PyObj = SWIG_NewPointerObj((void *) &sb_frame, SWIGTYPE_p_lldb__SBFrame, 0);
    PyObject *Bp_Loc_PyObj = SWIG_NewPointerObj ((void *) &sb_bp_loc, SWIGTYPE_p_lldb__SBBreakpointLocation, 0);
    
    if (Frame_PyObj == NULL || Bp_Loc_PyObj == NULL)
        return stop_at_breakpoint;
        
    if (!python_function_name || !session_dictionary_name)
        return stop_at_breakpoint;

    PyObject *session_dict, *pfunc;
    PyObject *pargs, *pvalue;
    
    session_dict = FindSessionDictionary (session_dictionary_name);
    if (session_dict != NULL)
    {
        pfunc = ResolvePythonName (python_function_name, session_dict);
        if (pfunc != NULL)
        {
            // Set up the arguments and call the function.
                
            if (PyCallable_Check (pfunc))
            {
                pargs = PyTuple_New (3);
                if (pargs == NULL)
                {
                    if (PyErr_Occurred())
                        PyErr_Clear();
                    return stop_at_breakpoint;
                }
                
                PyTuple_SetItem (pargs, 0, Frame_PyObj);  // This "steals" a reference to Frame_PyObj
                PyTuple_SetItem (pargs, 1, Bp_Loc_PyObj); // This "steals" a reference to Bp_Loc_PyObj
                PyTuple_SetItem (pargs, 2, session_dict); // This "steals" a reference to session_dict
                pvalue = PyObject_CallObject (pfunc, pargs);
                Py_DECREF (pargs);
                
                if (pvalue != NULL)
                {
                    Py_DECREF (pvalue);
                }
                else if (PyErr_Occurred ())
                {
                    PyErr_Clear();
                }
                Py_INCREF (session_dict);
            }
            else if (PyErr_Occurred())
            {
                PyErr_Clear();
            }
        }
        else if (PyErr_Occurred())
        {
            PyErr_Clear();
        }
    }
    else if (PyErr_Occurred ())
    {
        PyErr_Clear ();
    }
    return stop_at_breakpoint;
}

// This function is called by lldb_private::ScriptInterpreterPython::WatchpointCallbackFunction(...)
// and is used when a script command is attached to a watchpoint for execution.

SWIGEXPORT bool
LLDBSwigPythonWatchpointCallbackFunction 
(
    const char *python_function_name,
    const char *session_dictionary_name,
    const lldb::StackFrameSP& frame_sp, 
    const lldb::WatchpointSP& wp_sp
)
{
    lldb::SBFrame sb_frame (frame_sp);
    lldb::SBWatchpoint sb_wp(wp_sp);

    bool stop_at_watchpoint = true;
    PyObject *Frame_PyObj = SWIG_NewPointerObj((void *) &sb_frame, SWIGTYPE_p_lldb__SBFrame, 0);
    PyObject *Wp_PyObj = SWIG_NewPointerObj ((void *) &sb_wp, SWIGTYPE_p_lldb__SBWatchpoint, 0);
    
    if (Frame_PyObj == NULL || Wp_PyObj == NULL)
        return stop_at_watchpoint;
        
    if (!python_function_name || !session_dictionary_name)
        return stop_at_watchpoint;

    PyObject *session_dict, *pfunc;
    PyObject *pargs, *pvalue;
    
    session_dict = FindSessionDictionary (session_dictionary_name);
    if (session_dict != NULL)
    {
        pfunc = ResolvePythonName (python_function_name, session_dict);
        if (pfunc != NULL)
        {
            // Set up the arguments and call the function.
                
            if (PyCallable_Check (pfunc))
            {
                pargs = PyTuple_New (3);
                if (pargs == NULL)
                {
                    if (PyErr_Occurred())
                        PyErr_Clear();
                    return stop_at_watchpoint;
                }
                
                PyTuple_SetItem (pargs, 0, Frame_PyObj);  // This "steals" a reference to Frame_PyObj
                PyTuple_SetItem (pargs, 1, Wp_PyObj);     // This "steals" a reference to Wp_PyObj
                PyTuple_SetItem (pargs, 2, session_dict); // This "steals" a reference to session_dict
                pvalue = PyObject_CallObject (pfunc, pargs);
                Py_DECREF (pargs);
                
                if (pvalue != NULL)
                {
                    Py_DECREF (pvalue);
                }
                else if (PyErr_Occurred ())
                {
                    PyErr_Clear();
                }
                Py_INCREF (session_dict);
            }
            else if (PyErr_Occurred())
            {
                PyErr_Clear();
            }
        }
        else if (PyErr_Occurred())
        {
            PyErr_Clear();
        }
    }
    else if (PyErr_Occurred ())
    {
        PyErr_Clear ();
    }
    return stop_at_watchpoint;
}

SWIGEXPORT bool
LLDBSwigPythonCallTypeScript 
(
    const char *python_function_name,
    const void *session_dictionary,
    const lldb::ValueObjectSP& valobj_sp,
    void** pyfunct_wrapper,
    std::string& retval
)
{
    lldb::SBValue sb_value (valobj_sp);

    PyObject *ValObj_PyObj = SWIG_NewPointerObj((void *) &sb_value, SWIGTYPE_p_lldb__SBValue, 0);
    
    if (ValObj_PyObj == NULL)
        return false;
        
    if (!python_function_name || !session_dictionary)
        return false;

    PyObject *session_dict = (PyObject*)session_dictionary, *pfunc = NULL, *pargs = NULL, *pvalue = NULL;
    
    if (pyfunct_wrapper && *pyfunct_wrapper && PyFunction_Check (*pyfunct_wrapper))
    {
        pfunc = (PyObject*)(*pyfunct_wrapper);
        if (pfunc->ob_refcnt == 1)
        {
            Py_XDECREF(pfunc);
            pfunc = NULL;
        }
    }

    if (PyDict_Check(session_dict))
    {
        PyErr_Cleaner pyerr_cleanup(true);  // show Python errors
        
        if (!pfunc)
        {
            pfunc = ResolvePythonName (python_function_name, session_dict);
            if (!pfunc || !PyFunction_Check (pfunc))
                return false;
            else
            {
                if (pyfunct_wrapper)
                    *pyfunct_wrapper = pfunc;
            }
        }
        /*else
            printf("caching works!!!!\n");*/
        
        pargs = PyTuple_Pack(2, ValObj_PyObj, session_dict);
        if (pargs == NULL)
            return false;
        
        pvalue = PyObject_CallObject (pfunc, pargs);
        Py_DECREF (pargs);
        
        if (pvalue != NULL && pvalue != Py_None && PyString_Check(pvalue))
            retval.assign(PyString_AsString(pvalue));
        Py_XDECREF (pvalue);
        Py_INCREF (session_dict);
    }
    return true;
}

SWIGEXPORT void*
LLDBSwigPythonCreateSyntheticProvider 
(
    const std::string python_class_name,
    const char *session_dictionary_name,
    const lldb::ValueObjectSP& valobj_sp
)
{
    PyObject* retval = NULL;

    if (python_class_name.empty() || !session_dictionary_name)
        Py_RETURN_NONE;

    // I do not want the SBValue to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBValue *valobj_sb = new lldb::SBValue(valobj_sp);
    valobj_sb->SetPreferSyntheticValue(false);

    PyObject *ValObj_PyObj = SWIG_NewPointerObj((void *)valobj_sb, SWIGTYPE_p_lldb__SBValue, 0);

    if (ValObj_PyObj == NULL)
        Py_RETURN_NONE;

    const char* python_function_name = python_class_name.c_str();

    PyObject *session_dict, *pfunc;
    PyObject *pvalue;
    
    session_dict = FindSessionDictionary (session_dictionary_name);
    if (session_dict != NULL)
    {
        pfunc = ResolvePythonName (python_function_name, session_dict);
        if (pfunc != NULL)
        {
            // Set up the arguments and call the function.
                
            if (PyCallable_Check (pfunc))
            {
                PyObject *argList = Py_BuildValue("SS", ValObj_PyObj, session_dict);

                if (PyErr_Occurred ())
                {
                    PyErr_Print();
                    PyErr_Clear();
                    return retval;
                }

                if (argList == NULL)
                {
                    return retval;
                }

                Py_INCREF(ValObj_PyObj);

                pvalue = PyObject_CallObject(pfunc, argList);

                Py_DECREF(argList);

                if (pvalue != NULL)
                {
                    if (pvalue != Py_None)
                        retval = pvalue;
                    else
                    {
                        retval = Py_None;
                        Py_INCREF(retval);
                    }
                }
                else if (PyErr_Occurred ())
                {
                    PyErr_Print();
                    PyErr_Clear();
                }
                Py_INCREF (session_dict);
            }
            else if (PyErr_Occurred())
            {
                PyErr_Print();
                PyErr_Clear();
            }
        }
        else if (PyErr_Occurred())
        {
            PyErr_Print();
            PyErr_Clear();
        }
    }
    else if (PyErr_Occurred ())
    {
        PyErr_Print();
        PyErr_Clear ();
    }
    if (retval)
        return retval;
    else
        Py_RETURN_NONE;
}

// wrapper that calls an optional instance member of an object taking no arguments
static PyObject*
LLDBSwigPython_CallOptionalMember
(
    PyObject* self,
    char* callee_name,
    PyObject* ret_if_not_found = Py_None,
    bool* was_found = NULL
)
{
    if (self == NULL || self == Py_None)
    {
        if (was_found)
            *was_found = false;
        Py_XINCREF(ret_if_not_found);
        return ret_if_not_found;
    }

    PyObject* pmeth  = PyObject_GetAttrString(self, callee_name);

    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }

    if (pmeth == NULL || pmeth == Py_None)
    {
        if (was_found)
            *was_found = false;
        Py_XDECREF(pmeth);
        Py_XINCREF(ret_if_not_found);
        return ret_if_not_found;
    }

    if (PyCallable_Check(pmeth) == 0)
    {
        if (PyErr_Occurred())
        {
            PyErr_Clear();
        }

        Py_XDECREF(pmeth);
        if (was_found)
            *was_found = false;
        Py_XINCREF(ret_if_not_found);
        return ret_if_not_found;
    }

    if (was_found)
        *was_found = true;

    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }

    Py_XDECREF(pmeth);

    // right now we know this function exists and is callable..
    PyObject* py_return = PyObject_CallMethod(self, callee_name, NULL);
    
    // if it fails, print the error but otherwise go on
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    return py_return;
}

SWIGEXPORT uint32_t
LLDBSwigPython_CalculateNumChildren
(
    PyObject *implementor
)
{
    uint32_t ret_val = UINT32_MAX;

    static char callee_name[] = "num_children";

    PyObject* py_return = LLDBSwigPython_CallOptionalMember(implementor,callee_name, NULL);

    if (!py_return)
        return ret_val;

    if (PyInt_Check(py_return))
        ret_val = PyInt_AsLong(py_return);

    Py_XDECREF(py_return);
    
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    return ret_val;
}

SWIGEXPORT PyObject*
LLDBSwigPython_GetChildAtIndex
(
    PyObject *implementor,
    uint32_t idx
)
{

    static char callee_name[] = "get_child_at_index";
    static char param_format[] = "i";

    if (implementor == NULL || implementor == Py_None)
        return NULL;
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, param_format, idx);
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    if (py_return == NULL || py_return == Py_None)
    {
        Py_XDECREF(py_return);
        return NULL;
    }
    
    lldb::SBValue* sbvalue_ptr = NULL;

    if (SWIG_ConvertPtr(py_return, (void**)&sbvalue_ptr, SWIGTYPE_p_lldb__SBValue, 0) == -1)
    {
        Py_DECREF(py_return);
        return NULL;
    }
    
    if (sbvalue_ptr == NULL)
        return NULL;

    return py_return;
}

SWIGEXPORT int
LLDBSwigPython_GetIndexOfChildWithName
(
    PyObject *implementor,
    const char* child_name
)
{
    static char callee_name[] = "get_child_index";
    static char param_format[] = "s";

    if (implementor == NULL || implementor == Py_None)
        return 0;
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, param_format, child_name);
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    if (py_return == NULL || py_return == Py_None)
    {
        Py_XDECREF(py_return);
        return UINT32_MAX;
    }
    long retval = PyInt_AsLong(py_return);
    Py_DECREF(py_return);
    if (retval >= 0)
        return (uint32_t)retval;
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    return 0;
}

SWIGEXPORT bool
LLDBSwigPython_UpdateSynthProviderInstance
(
    PyObject *implementor
)
{
    bool ret_val = false;

    static char callee_name[] = "update";

    PyObject* py_return = LLDBSwigPython_CallOptionalMember(implementor,callee_name);

    if (py_return == Py_True)
        ret_val = true;

    Py_XDECREF(py_return);
    
    return ret_val;
}

SWIGEXPORT bool
LLDBSwigPython_MightHaveChildrenSynthProviderInstance
(
    PyObject *implementor
)
{
    bool ret_val = false;

    static char callee_name[] = "has_children";

    PyObject* py_return = LLDBSwigPython_CallOptionalMember(implementor,callee_name, Py_True);

    if (py_return == Py_True)
        ret_val = true;

    Py_XDECREF(py_return);
    
    return ret_val;
}

SWIGEXPORT void*
LLDBSWIGPython_CastPyObjectToSBValue
(
    PyObject* data
)
{
    lldb::SBValue* sb_ptr = NULL;
    
    int valid_cast = SWIG_ConvertPtr(data, (void**)&sb_ptr, SWIGTYPE_p_lldb__SBValue, 0);

    if (valid_cast == -1)
        return NULL;

    return sb_ptr;
}

// Currently, SBCommandReturnObjectReleaser wraps an std::auto_ptr to an
// lldb_private::CommandReturnObject. This means that the destructor for the
// SB object will deallocate its contained CommandReturnObject. Because that
// object is used as the real return object for Python-based commands, we want
// it to stay around. Thus, we release the auto_ptr before returning from
// LLDBSwigPythonCallCommand, and to guarantee that the release will occur no
// matter how we exit from the function, we have a releaser object whose
// destructor does the right thing for us
class SBCommandReturnObjectReleaser
{
public:
    SBCommandReturnObjectReleaser (lldb::SBCommandReturnObject &obj) :
        m_command_return_object_ref (obj)
    {
    }

    ~SBCommandReturnObjectReleaser ()
    {
        m_command_return_object_ref.Release();
    }
private:
    lldb::SBCommandReturnObject &m_command_return_object_ref;
};

SWIGEXPORT bool
LLDBSwigPythonCallCommand 
(
    const char *python_function_name,
    const char *session_dictionary_name,
    lldb::DebuggerSP& debugger,
    const char* args,
    std::string& err_msg,
    lldb_private::CommandReturnObject& cmd_retobj
)
{

    lldb::SBCommandReturnObject cmd_retobj_sb(&cmd_retobj);
    SBCommandReturnObjectReleaser cmd_retobj_sb_releaser(cmd_retobj_sb);
    lldb::SBDebugger debugger_sb(debugger);

    bool retval = false;

    PyObject *DebuggerObj_PyObj = SWIG_NewPointerObj((void *) &debugger_sb, SWIGTYPE_p_lldb__SBDebugger, 0);
    PyObject *CmdRetObj_PyObj = SWIG_NewPointerObj((void *) &cmd_retobj_sb, SWIGTYPE_p_lldb__SBCommandReturnObject, 0);

    if (DebuggerObj_PyObj == NULL)
        return retval;
        
    if (CmdRetObj_PyObj == NULL)
        return retval;

    if (!python_function_name || !session_dictionary_name)
        return retval;

    PyObject *session_dict, *pfunc;
    PyObject *pargs, *pvalue;
    
    session_dict = FindSessionDictionary (session_dictionary_name);
    if (session_dict != NULL)
    {
        pfunc = ResolvePythonName (python_function_name, session_dict);
        if (pfunc != NULL)
        {
            // Set up the arguments and call the function.
                
            if (PyCallable_Check (pfunc))
            {
                pargs = PyTuple_New (4);
                if (pargs == NULL)
                {
                    if (PyErr_Occurred())
                        PyErr_Clear();
                    return retval;
                }
                
                PyTuple_SetItem (pargs, 0, DebuggerObj_PyObj);  // This "steals" a reference to DebuggerObj_PyObj
                PyTuple_SetItem (pargs, 1, PyString_FromString(args));
                PyTuple_SetItem (pargs, 2, CmdRetObj_PyObj);  // This "steals" a reference to CmdRetObj_PyObj
                PyTuple_SetItem (pargs, 3, session_dict); // This "steals" a reference to session_dict
                pvalue = PyObject_CallObject (pfunc, pargs);
                Py_DECREF (pargs);
                
                if (pvalue != NULL)
                {
                    if (pvalue == Py_None) // no error
                    {
                        err_msg.clear();
                        retval = true;
                    }
                    else
                    {
                        // return value is an error string
                        if (PyString_CheckExact(pvalue))
                            err_msg.assign(PyString_AsString(pvalue));
                        retval = false;
                    }
                    Py_DECREF (pvalue);
                }
                else if (PyErr_Occurred ())
                {
                    PyErr_Print();
                    PyErr_Clear();
                }
                Py_INCREF (session_dict);
            }
            else if (PyErr_Occurred())
            {
                PyErr_Print();
                PyErr_Clear();
            }
        }
        else if (PyErr_Occurred())
        {
            PyErr_Print();
            PyErr_Clear();
        }
    }
    else if (PyErr_Occurred ())
    {
        PyErr_Print();
        PyErr_Clear ();
    }
    return retval;
}

SWIGEXPORT void*
LLDBSWIGPythonCreateOSPlugin
(
    const std::string python_class_name,
    const char *session_dictionary_name,
    const lldb::ProcessSP& process_sp
)
{
    PyObject* retval = NULL;

    if (python_class_name.empty() || !session_dictionary_name)
        Py_RETURN_NONE;

    // I do not want the SBValue to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBProcess *process_sb = new lldb::SBProcess(process_sp);

    PyObject *SBProc_PyObj = SWIG_NewPointerObj((void *)process_sb, SWIGTYPE_p_lldb__SBProcess, 0);

    if (SBProc_PyObj == NULL)
        Py_RETURN_NONE;

    const char* python_function_name = python_class_name.c_str();

    PyObject *session_dict, *pfunc;
    PyObject *pvalue;
    
    session_dict = FindSessionDictionary (session_dictionary_name);
    if (session_dict != NULL)
    {
        pfunc = ResolvePythonName (python_function_name, session_dict);
        if (pfunc != NULL)
        {
            // Set up the arguments and call the function.
                
            if (PyCallable_Check (pfunc))
            {
                PyObject *argList = Py_BuildValue("(O)", SBProc_PyObj);

                if (PyErr_Occurred ())
                {
                    PyErr_Print();
                    PyErr_Clear();
                    return retval;
                }

                if (argList == NULL)
                {
                    return retval;
                }

                Py_INCREF(SBProc_PyObj);

                pvalue = PyObject_CallObject(pfunc, argList);

                Py_DECREF(argList);

                if (pvalue != NULL)
                {
                    if (pvalue != Py_None)
                        retval = pvalue;
                    else
                    {
                        retval = Py_None;
                        Py_INCREF(retval);
                    }
                }
                else if (PyErr_Occurred ())
                {
                    PyErr_Print();
                    PyErr_Clear();
                }
                Py_INCREF (session_dict);
            }
            else if (PyErr_Occurred())
            {
                PyErr_Print();
                PyErr_Clear();
            }
        }
        else if (PyErr_Occurred())
        {
            PyErr_Print();
            PyErr_Clear();
        }
    }
    else if (PyErr_Occurred ())
    {
        PyErr_Print();
        PyErr_Clear ();
    }
    if (retval)
        return retval;
    else
        Py_RETURN_NONE;
}

SWIGEXPORT bool
LLDBSwigPythonCallModuleInit 
(
    const std::string python_module_name,
    const char *session_dictionary_name,
    lldb::DebuggerSP& debugger
)
{

    lldb::SBDebugger debugger_sb(debugger);

    bool retval = false;

    PyObject *DebuggerObj_PyObj = SWIG_NewPointerObj((void *) &debugger_sb, SWIGTYPE_p_lldb__SBDebugger, 0);

    if (DebuggerObj_PyObj == NULL)
        return retval;
        
    if (!(python_module_name.length()) || !session_dictionary_name)
        return retval;

    PyObject *session_dict, *pfunc;
    PyObject *pargs, *pvalue;
    
    session_dict = FindSessionDictionary (session_dictionary_name);
    
    std::string python_function_name_string = python_module_name + (".__lldb_init_module");
    const char* python_function_name = python_function_name_string.c_str();
    
    if (session_dict != NULL)
    {
        pfunc = ResolvePythonName (python_function_name, session_dict);
        
        if (PyErr_Occurred()) // this might not exist.. let's make sure we handle that
        {
            PyErr_Clear();
            return true;
        }

        if (pfunc == NULL)
            return true;
        else
        {
            // Set up the arguments and call the function.
                
            if (PyCallable_Check (pfunc))
            {
                pargs = PyTuple_New (2);
                if (pargs == NULL)
                {
                    if (PyErr_Occurred())
                        PyErr_Clear();
                    return retval;
                }
                
                PyTuple_SetItem (pargs, 0, DebuggerObj_PyObj);  // This "steals" a reference to DebuggerObj_PyObj
                PyTuple_SetItem (pargs, 1, session_dict); // This "steals" a reference to session_dict
                pvalue = PyObject_CallObject (pfunc, pargs);
                Py_DECREF (pargs);
                
                if (PyErr_Occurred ())
                {
                    PyErr_Print();
                    PyErr_Clear();
                }
                else
                {
                    retval = true;
                    Py_XDECREF (pvalue);
                }
                Py_INCREF (session_dict);
            }
            else if (PyErr_Occurred())
            {
                PyErr_Print();
                PyErr_Clear();
            }
        }
    }
    return retval;
}
%}


%runtime %{
// Forward declaration to be inserted at the start of LLDBWrapPython.h
// I used runtime as a hack to make SWIG place it where it's needed.
// This is needed to use LLDBSwigPythonCallSBInputReaderCallback in the
// typemaps and in the extensions (SBInputReader.__del__()).
#include "lldb/API/SBInputReader.h"
#include "lldb/API/SBDebugger.h"

size_t
LLDBSwigPythonCallSBInputReaderCallback(void *baton,
                                        lldb::SBInputReader *reader,
                                        lldb::InputReaderAction notification,
                                        const char*bytes,
                                        size_t bytes_len);

void LLDBSwigPythonCallPythonLogOutputCallback(const char *str, void *baton);
%}

%wrapper %{
// For the InputReader Callback functions
SWIGEXPORT size_t
LLDBSwigPythonCallSBInputReaderCallback(void *baton,
                                        lldb::SBInputReader *reader,
                                        lldb::InputReaderAction notification,
                                        const char*bytes,
                                        size_t bytes_len) {
    if (baton != Py_None) {
        SWIG_PYTHON_THREAD_BEGIN_BLOCK;
    
        PyObject *py_InputReader = SWIG_NewPointerObj(reader, SWIGTYPE_p_lldb__SBInputReader, false);
        PyObject *py_Notification = PyInt_FromLong(notification);
        PyObject *py_Bytes = PyBytes_FromStringAndSize(bytes, bytes_len);
    
        PyObject *tuple = PyTuple_Pack(3, py_InputReader, py_Notification, py_Bytes);
        PyObject *res = PyObject_Call(reinterpret_cast<PyObject*>(baton), tuple, NULL);
        Py_DECREF(tuple);
        Py_DECREF(py_InputReader);
        Py_DECREF(py_Notification);
        Py_DECREF(py_Bytes);
    
        if (res == NULL) {
          PyObject *exc = PyErr_Occurred();
          if (exc) {
            ::puts("\nErroring out at LLDBSwigPythonCallSBInputReaderCallback");
            PyErr_Print();
          }
          return 0;
        }
    
        size_t result = 0;
        // If the callback misbehaves and returns Py_None, assume it returned 0
        if (res != Py_None)
          result = static_cast<size_t>(PyInt_AsSsize_t(res));
    
        Py_DECREF(res);
        SWIG_PYTHON_THREAD_END_BLOCK;
        return result;
    }
    return 0;
}

// For the LogOutputCallback functions
void LLDBSwigPythonCallPythonLogOutputCallback(const char *str, void *baton) {
    if (baton != Py_None) {
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      PyObject_CallFunction(reinterpret_cast<PyObject*>(baton), const_cast<char*>("s"), str);
      SWIG_PYTHON_THREAD_END_BLOCK;
    }
}
%}
