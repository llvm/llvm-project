%header %{

template <typename T>
PyObject *
SBTypeToSWIGWrapper (T* item);

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
            if(m_print && !PyErr_ExceptionMatches(PyExc_SystemExit))
                PyErr_Print();
            PyErr_Clear();
        }
    }

private:
    bool m_print;
};

// TODO(zturner): This entire class should be moved to PythonDataObjects.h
// and properly abstracted and unit-tested.
class PyCallable
{
public:
    struct argc {
        size_t num_args;
        bool varargs : 1;
        bool kwargs : 1;
    };

    argc
    GetNumArguments ()
    {
        PyObject *py_func_obj = NULL;
        if (m_callable)
        {
            if (PyMethod_Check(m_callable))
                py_func_obj = PyMethod_GET_FUNCTION(m_callable);
            else
                py_func_obj = m_callable;
        }

        if (py_func_obj)
        {
            PyCodeObject* code = (PyCodeObject*)PyFunction_GET_CODE(py_func_obj);
            if (code)
            {
                size_t args = code->co_argcount;
                bool va=false,kw=false;
                if ((code->co_flags & 4) == 4)
                    va = true;
                if ((code->co_flags & 8) == 8)
                    kw = true;
                return {args,va,kw};
            }
        }
        return {SIZE_MAX,false,false};
    }

    operator
    bool ()
    {
        return m_callable != NULL;
    }

    template<typename ...Args>
    PyObject*
    operator () (Args... args)
    {
        return (*this)({SBTypeToSWIGWrapper(args)...});
    }

    PyObject*
    operator () (std::initializer_list<PyObject*> args)
    {
        PyObject* retval = NULL;
        PyObject* pargs = PyTuple_New (args.size());
        if (pargs == NULL)
        {
            if (PyErr_Occurred())
                PyErr_Clear();
            return retval;
        }
        size_t idx = 0;
        for (auto arg : args)
        {
            if (!arg)
                return retval;
            Py_INCREF(arg); // _SetItem steals a reference
            PyTuple_SetItem(pargs,idx,arg);
            idx++;
        }
        retval = PyObject_CallObject (m_callable, pargs);
        Py_XDECREF (pargs);
        return retval;
    }

    static PyCallable
    FindWithPythonObject (PyObject* pfunc)
    {
        return PyCallable(pfunc);
    }

    static PyCallable
    FindWithFunctionName (const char *python_function_name,
                          const char *session_dictionary_name)
    {
        if (!python_function_name || !session_dictionary_name)
            return PyCallable();
        if ( (python_function_name[0] == 0) || (session_dictionary_name[0] == 0) )
            return PyCallable();

        using namespace lldb_private;
        auto dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
        return FindWithFunctionName(python_function_name, dict.get());
    }

    static PyCallable
    FindWithFunctionName (const char *python_function_name,
                          PyObject *py_session_dict)
    {
        if (!python_function_name || !py_session_dict)
            return PyCallable();
        if ( (python_function_name[0] == 0))
            return PyCallable();

        using namespace lldb_private;
        PythonDictionary session_dict(PyRefType::Borrowed, py_session_dict);
        PythonCallable result = PythonObject::ResolveNameWithDictionary(python_function_name, session_dict).AsType<PythonCallable>();
        return PyCallable(result.release());
    }

    static PyCallable
    FindWithMemberFunction (PyObject *self,
                            const char *python_function_name)
    {
        if (self == NULL || self == Py_None)
            return PyCallable();
        if (!python_function_name || (python_function_name[0] == 0))
            return PyCallable();

        return PyCallable(PyObject_GetAttrString(self, python_function_name));
    }

private:
    PyObject* m_callable;

    PyCallable (PyObject *callable = NULL) :
    m_callable(callable)
    {
        if (m_callable && PyCallable_Check(m_callable) == false)
            m_callable = NULL;
    }
};

%}

%wrapper %{

// resolve a dotted Python name in the form
// foo.bar.baz.Foobar to an actual Python object
// if pmodule is NULL, the __main__ module will be used
// as the starting point for the search


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
    using namespace lldb_private;
    lldb::SBFrame sb_frame (frame_sp);
    lldb::SBBreakpointLocation sb_bp_loc(bp_loc_sp);

    bool stop_at_breakpoint = true;

    PyErr_Cleaner py_err_cleaner(true);
    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();

    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return stop_at_breakpoint;

    PyObject* pvalue = NULL;
    pvalue = pfunc(sb_frame, sb_bp_loc, dict.get());

    if (pvalue == Py_False)
        stop_at_breakpoint = false;

    Py_XDECREF (pvalue);

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
    using namespace lldb_private;
    lldb::SBFrame sb_frame (frame_sp);
    lldb::SBWatchpoint sb_wp(wp_sp);

    bool stop_at_watchpoint = true;

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();

    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return stop_at_watchpoint;

    PyObject* pvalue = NULL;
    pvalue = pfunc(sb_frame, sb_wp, dict.get());

    if (pvalue == Py_False)
        stop_at_watchpoint = false;

    Py_XDECREF (pvalue);

    return stop_at_watchpoint;
}

bool
PyObjectToString (PyObject* object,
                  std::string& retval)
{
    retval.clear();
    bool was_ok = false;
    if (object != NULL && object != Py_None)
    {
        if (PyString_Check(object))
        {
            retval.assign(PyString_AsString(object));
            was_ok = true;
        }
        else
        {
            PyObject* value_as_string = PyObject_Str(object);
            if (value_as_string && value_as_string != Py_None && PyString_Check(value_as_string))
            {
                retval.assign(PyString_AsString(value_as_string));
                was_ok = true;
            }
            Py_XDECREF(value_as_string);
        }
    }
    return was_ok;
}

SWIGEXPORT bool
LLDBSwigPythonCallTypeScript
(
    const char *python_function_name,
    const void *session_dictionary,
    const lldb::ValueObjectSP& valobj_sp,
    void** pyfunct_wrapper,
    const lldb::TypeSummaryOptionsSP& options_sp,
    std::string& retval
)
{
    using namespace lldb_private;
    lldb::SBValue sb_value (valobj_sp);
    lldb::SBTypeSummaryOptions sb_options(options_sp.get());

    retval.clear();

    if (!python_function_name || !session_dictionary)
        return false;

    PyObject *pfunc_impl = NULL, *pvalue = NULL;

    if (pyfunct_wrapper && *pyfunct_wrapper && PyFunction_Check (*pyfunct_wrapper))
    {
        pfunc_impl = (PyObject*)(*pyfunct_wrapper);
        if (pfunc_impl->ob_refcnt == 1)
        {
            Py_XDECREF(pfunc_impl);
            pfunc_impl = NULL;
        }
    }

    PyObject *py_dict = (PyObject*)session_dictionary;
    if (!PythonDictionary::Check(py_dict))
        return true;

    PythonDictionary dict(PyRefType::Borrowed, py_dict);

    PyErr_Cleaner pyerr_cleanup(true);  // show Python errors

    if (!pfunc_impl)
    {
        pfunc_impl = PythonObject::ResolveNameWithDictionary(python_function_name, dict).release();
        if (!pfunc_impl || !PyCallable_Check (pfunc_impl))
            return false;
        else
        {
            if (pyfunct_wrapper)
                *pyfunct_wrapper = pfunc_impl;
        }
    }

    PyCallable pfunc = PyCallable::FindWithPythonObject(pfunc_impl);

    if (!pfunc)
        return false;

    // if the third argument is supported, or varargs are allowed
    PyCallable::argc argc = pfunc.GetNumArguments();
    if (argc.num_args == 3 || argc.varargs == true)
        pvalue = pfunc(sb_value,dict.get(),sb_options);
    else
        pvalue = pfunc(sb_value,dict.get());

    PyObjectToString(pvalue,retval);

    Py_XDECREF (pvalue);
    return true;
}

SWIGEXPORT void*
LLDBSwigPythonCreateSyntheticProvider
(
    const char *python_class_name,
    const char *session_dictionary_name,
    const lldb::ValueObjectSP& valobj_sp
)
{
    using namespace lldb_private;
    PyObject* retval = NULL;

    if (python_class_name == NULL || python_class_name[0] == '\0' || !session_dictionary_name)
        Py_RETURN_NONE;

    // I do not want the SBValue to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBValue *sb_value = new lldb::SBValue(valobj_sp);
    sb_value->SetPreferSyntheticValue(false);
    PyObject *ValObj_PyObj = SBTypeToSWIGWrapper(sb_value);

    if (ValObj_PyObj == NULL)
        Py_RETURN_NONE;

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_class_name,dict.get());

    if (!pfunc)
        return retval;

    Py_INCREF(ValObj_PyObj);

    retval = pfunc(sb_value, dict.get());

    Py_XINCREF(retval);

    if (retval)
        return retval;
    else
        Py_RETURN_NONE;
}

SWIGEXPORT void*
LLDBSwigPythonCreateCommandObject
(
    const char *python_class_name,
    const char *session_dictionary_name,
    const lldb::DebuggerSP debugger_sp
)
{
    using namespace lldb_private;
    PyObject* retval = NULL;

    if (python_class_name == NULL || python_class_name[0] == '\0' || !session_dictionary_name)
        Py_RETURN_NONE;

    lldb::SBDebugger debugger_sb(debugger_sp);

    PyErr_Cleaner py_err_cleaner(true);
    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_class_name,dict.get());

    if (!pfunc)
        return retval;

    retval = pfunc(debugger_sb, dict.get());

    Py_XINCREF(retval);

    if (retval)
        return retval;
    else
        Py_RETURN_NONE;
}

SWIGEXPORT void*
LLDBSwigPythonCreateScriptedThreadPlan
(
    const char *python_class_name,
    const char *session_dictionary_name,
    const lldb::ThreadPlanSP& thread_plan_sp
)
{
    using namespace lldb_private;
    PyObject* retval = NULL;

    if (python_class_name == NULL || python_class_name[0] == '\0' || !session_dictionary_name)
        Py_RETURN_NONE;

    // I do not want the SBThreadPlan to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBThreadPlan *tp_value = new lldb::SBThreadPlan(thread_plan_sp);

    PyObject *ThreadPlan_PyObj = SBTypeToSWIGWrapper(tp_value);

    if (ThreadPlan_PyObj == NULL)
        Py_RETURN_NONE;

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_class_name, dict.get());

    if (!pfunc)
        return retval;

    Py_INCREF(ThreadPlan_PyObj);

    retval = pfunc(tp_value, dict.get());

    // FIXME: At this point we should check that the class we found supports all the methods
    // that we need.

    Py_XINCREF(retval);

    if (retval)
        return retval;
    else
        Py_RETURN_NONE;
}

SWIGEXPORT bool
LLDBSWIGPythonCallThreadPlan
(
    void *implementor,
    const char *method_name,
    lldb_private::Event *event,
    bool &got_error
)
{
    bool ret_val = false;
    got_error = false;


    PyErr_Cleaner py_err_cleaner(false);

    PyCallable pfunc = PyCallable::FindWithMemberFunction((PyObject *) implementor, method_name);

    if (!pfunc)
    {
        return ret_val;
    }

    PyObject* py_return = Py_None;

    if (event != NULL)
    {
        lldb::SBEvent sb_event(event);

        PyObject *py_obj_event = SBTypeToSWIGWrapper(sb_event);

        py_return = pfunc(py_obj_event);
    }
    else
    {
        py_return = pfunc();
    }

    if (PyErr_Occurred())
    {
        got_error = true;
        printf ("Return value was neither false nor true for call to %s.\n", method_name);
        PyErr_Print();
    }
    else
    {
        if (py_return == Py_True)
            ret_val = true;
        else if (py_return == Py_False)
            ret_val = false;
        else
        {
            // Somebody returned the wrong thing...
            got_error = true;
            printf ("Wrong return value type for call to %s.\n", method_name);
        }
    }

    Py_XDECREF(py_return);

    return ret_val;
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
    PyErr_Cleaner py_err_cleaner(false);

    PyCallable pfunc = PyCallable::FindWithMemberFunction(self,callee_name);

    if (!pfunc)
    {
        if (was_found)
            *was_found = false;
        Py_XINCREF(ret_if_not_found);
        return ret_if_not_found;
    }

    if (was_found)
        *was_found = true;

    PyObject* py_return = pfunc();
    return py_return;
}

SWIGEXPORT size_t
LLDBSwigPython_CalculateNumChildren
(
    PyObject *implementor,
    uint32_t max
)
{
    size_t ret_val = 0;
    bool int_match = false;

    PyCallable pfunc = PyCallable::FindWithMemberFunction(implementor, "num_children");

    if (!pfunc)
        return ret_val;

    PyObject* py_return = NULL;
    auto argc = pfunc.GetNumArguments();
    if (argc.num_args == 1)
        py_return = pfunc();
    else if (argc.num_args == 2)
        py_return = pfunc(max);

    if (!py_return)
        return ret_val;

    // PyInt_* are not available for Python 3 and above.
#if PY_MAJOR_VERSION < 3
    if (PyInt_Check (py_return))
    {
        int_match = true;
        ret_val = static_cast<size_t> (PyInt_AsLong (py_return));
    }
#endif

    // We want to check for PyLong only if the return value did not
    // match PyInt. This is because we do not want to call PyLong_Check if
    // PyInt_Check returns true but PyInt_AsLong generates an error.
    if (!int_match && PyLong_Check (py_return))
    {
#if PY_MAJOR_VERSION < 3
        ret_val = static_cast<size_t> (PyLong_AsUnsignedLong (py_return));
#else
        // PyLong_AsSize_t is available only for Python 3 and above.
        ret_val = PyLong_AsSize_t (py_return);
#endif
    }

    Py_XDECREF(py_return);

    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }

    if (argc.num_args == 1 && ret_val > max)
        ret_val = max;

    return ret_val;
}

SWIGEXPORT PyObject*
LLDBSwigPython_GetChildAtIndex
(
    PyObject *implementor,
    uint32_t idx
)
{
    PyErr_Cleaner py_err_cleaner(true);

    PyCallable pfunc = PyCallable::FindWithMemberFunction(implementor,"get_child_at_index");

    if (!pfunc)
        return NULL;

    PyObject *py_return = NULL;
    py_return = pfunc(idx);

    if (py_return == NULL || py_return == Py_None)
    {
        Py_XDECREF(py_return);
        return NULL;
    }

    lldb::SBValue* sbvalue_ptr = NULL;

    if (SWIG_ConvertPtr(py_return, (void**)&sbvalue_ptr, SWIGTYPE_p_lldb__SBValue, 0) == -1)
    {
        Py_XDECREF(py_return);
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
    PyErr_Cleaner py_err_cleaner(true);

    PyCallable pfunc = PyCallable::FindWithMemberFunction(implementor,"get_child_index");

    if (!pfunc)
        return UINT32_MAX;

    PyObject *py_return = NULL;
    py_return = pfunc(child_name);

    if (py_return == NULL || py_return == Py_None)
    {
        Py_XDECREF(py_return);
        return UINT32_MAX;
    }

    long retval = PyInt_AsLong(py_return);
    Py_XDECREF(py_return);

    if (retval >= 0)
        return (uint32_t)retval;

    return UINT32_MAX;
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

SWIGEXPORT PyObject*
LLDBSwigPython_GetValueSynthProviderInstance
(
    PyObject *implementor
)
{
    PyObject* ret_val = nullptr;

    static char callee_name[] = "get_value";

    PyObject* py_return = LLDBSwigPython_CallOptionalMember(implementor,callee_name, Py_None);

    if (py_return == Py_None || py_return == nullptr)
        ret_val = nullptr;

    lldb::SBValue* sbvalue_ptr = NULL;

    if (SWIG_ConvertPtr(py_return, (void**)&sbvalue_ptr, SWIGTYPE_p_lldb__SBValue, 0) == -1)
        ret_val = nullptr;
    else if (sbvalue_ptr == NULL)
        ret_val = nullptr;
    else
        ret_val = py_return;

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

// Currently, SBCommandReturnObjectReleaser wraps a unique pointer to an
// lldb_private::CommandReturnObject. This means that the destructor for the
// SB object will deallocate its contained CommandReturnObject. Because that
// object is used as the real return object for Python-based commands, we want
// it to stay around. Thus, we release the unique pointer before returning from
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
    lldb_private::CommandReturnObject& cmd_retobj,
    lldb::ExecutionContextRefSP exe_ctx_ref_sp
)
{
    using namespace lldb_private;
    lldb::SBCommandReturnObject cmd_retobj_sb(&cmd_retobj);
    SBCommandReturnObjectReleaser cmd_retobj_sb_releaser(cmd_retobj_sb);
    lldb::SBDebugger debugger_sb(debugger);
    lldb::SBExecutionContext exe_ctx_sb(exe_ctx_ref_sp);

    bool retval = false;

    PyErr_Cleaner py_err_cleaner(true);
    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    // pass the pointer-to cmd_retobj_sb or watch the underlying object disappear from under you
    // see comment above for SBCommandReturnObjectReleaser for further details
    PyObject* pvalue = NULL;

    PyCallable::argc argc = pfunc.GetNumArguments();
    if (argc.num_args == 5 || argc.varargs == true)
        pvalue = pfunc(debugger_sb, args, exe_ctx_sb, &cmd_retobj_sb, dict.get());
    else
        pvalue = pfunc(debugger_sb, args, &cmd_retobj_sb, dict.get());

    Py_XDECREF (pvalue);

    retval = true;

    return retval;
}

SWIGEXPORT bool
LLDBSwigPythonCallCommandObject
(
    PyObject *implementor,
    lldb::DebuggerSP& debugger,
    const char* args,
    lldb_private::CommandReturnObject& cmd_retobj,
    lldb::ExecutionContextRefSP exe_ctx_ref_sp
)
{

    lldb::SBCommandReturnObject cmd_retobj_sb(&cmd_retobj);
    SBCommandReturnObjectReleaser cmd_retobj_sb_releaser(cmd_retobj_sb);
    lldb::SBDebugger debugger_sb(debugger);
    lldb::SBExecutionContext exe_ctx_sb(exe_ctx_ref_sp);

    PyErr_Cleaner py_err_cleaner(true);

    PyCallable pfunc = PyCallable::FindWithMemberFunction(implementor,"__call__");

    if (!pfunc)
        return false;

    // pass the pointer-to cmd_retobj_sb or watch the underlying object disappear from under you
    // see comment above for SBCommandReturnObjectReleaser for further details
    PyObject* pvalue = NULL;

    pvalue = pfunc(debugger_sb, args, exe_ctx_sb, &cmd_retobj_sb);

    Py_XDECREF (pvalue);

    return true;
}

SWIGEXPORT void*
LLDBSWIGPythonCreateOSPlugin
(
    const char *python_class_name,
    const char *session_dictionary_name,
    const lldb::ProcessSP& process_sp
)
{
    using namespace lldb_private;
    PyObject* retval = NULL;

    if (python_class_name == NULL || python_class_name[0] == '\0' || !session_dictionary_name)
        Py_RETURN_NONE;

    // I do not want the SBProcess to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBProcess *process_sb = new lldb::SBProcess(process_sp);

    PyObject *SBProc_PyObj = SBTypeToSWIGWrapper(process_sb);

    if (SBProc_PyObj == NULL)
        Py_RETURN_NONE;

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_class_name,dict.get());

    if (!pfunc)
        return retval;

    Py_INCREF(SBProc_PyObj);

    retval = pfunc(SBProc_PyObj);

    Py_XINCREF(retval);

    if (retval)
        return retval;
    else
        Py_RETURN_NONE;
}

SWIGEXPORT void*
LLDBSWIGPython_GetDynamicSetting (void* module, const char* setting, const lldb::TargetSP& target_sp)
{

    if (!module || !setting)
        Py_RETURN_NONE;

    lldb::SBTarget target_sb(target_sp);

    PyObject *pvalue = NULL;

    {
        PyErr_Cleaner py_err_cleaner(true);
        PyCallable pfunc = PyCallable::FindWithFunctionName("get_dynamic_setting",(PyObject *)module);

        if (!pfunc)
            Py_RETURN_NONE;

        pvalue = pfunc(target_sb, setting);
    }

    return pvalue;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordProcess
(const char* python_function_name,
const char* session_dictionary_name,
lldb::ProcessSP& process,
std::string& output)

{
    using namespace lldb_private;
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBProcess process_sb(process);

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    PyObject* pvalue = NULL;
    pvalue = pfunc(process_sb, dict.get());

    if (PyObjectToString(pvalue,output))
        retval = true;

    Py_XDECREF(pvalue);

    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordThread
(const char* python_function_name,
const char* session_dictionary_name,
lldb::ThreadSP& thread,
std::string& output)

{
    using namespace lldb_private;
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBThread thread_sb(thread);

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    PyObject* pvalue = NULL;
    pvalue = pfunc(thread_sb, dict.get());

    if (PyObjectToString(pvalue,output))
        retval = true;

    Py_XDECREF(pvalue);

    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordTarget
(const char* python_function_name,
const char* session_dictionary_name,
lldb::TargetSP& target,
std::string& output)

{
    using namespace lldb_private;
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBTarget target_sb(target);

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    PyObject* pvalue = NULL;
    pvalue = pfunc(target_sb, dict.get());

    if (PyObjectToString(pvalue,output))
        retval = true;

    Py_XDECREF(pvalue);

    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordFrame
(const char* python_function_name,
const char* session_dictionary_name,
lldb::StackFrameSP& frame,
std::string& output)

{
    using namespace lldb_private;
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBFrame frame_sb(frame);

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    PyObject* pvalue = NULL;
    pvalue = pfunc(frame_sb, dict.get());

    if (PyObjectToString(pvalue,output))
        retval = true;

    Py_XDECREF(pvalue);

    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordValue
(const char* python_function_name,
const char* session_dictionary_name,
lldb::ValueObjectSP& value,
std::string& output)

{
    using namespace lldb_private;
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBValue value_sb(value);

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    PyObject* pvalue = NULL;
    pvalue = pfunc(value_sb, dict.get());

    if (PyObjectToString(pvalue,output))
        retval = true;

    Py_XDECREF(pvalue);

    return retval;
}

SWIGEXPORT bool
LLDBSwigPythonCallModuleInit
(
    const char *python_module_name,
    const char *session_dictionary_name,
    lldb::DebuggerSP& debugger
)
{
    using namespace lldb_private;
    bool retval = false;

    lldb::SBDebugger debugger_sb(debugger);

    std::string python_function_name_string = python_module_name;
    python_function_name_string += ".__lldb_init_module";
    const char* python_function_name = python_function_name_string.c_str();

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return true;

    PyObject* pvalue = NULL;
    pvalue = pfunc(debugger_sb, dict.get());

    retval = true;

    Py_XDECREF(pvalue);

    return retval;
}
%}


%runtime %{
// Forward declaration to be inserted at the start of LLDBWrapPython.h
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBValue.h"

SWIGEXPORT lldb::ValueObjectSP
LLDBSWIGPython_GetValueObjectSPFromSBValue (void* data)
{
    lldb::ValueObjectSP valobj_sp;
    if (data)
    {
        lldb::SBValue* sb_ptr = (lldb::SBValue *)data;
        valobj_sp = sb_ptr->GetSP();
    }
    return valobj_sp;
}

#ifdef __cplusplus
extern "C" {
#endif

void LLDBSwigPythonCallPythonLogOutputCallback(const char *str, void *baton);

#ifdef __cplusplus
}
#endif
%}

%wrapper %{


// For the LogOutputCallback functions
void LLDBSwigPythonCallPythonLogOutputCallback(const char *str, void *baton) {
    if (baton != Py_None) {
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      PyObject_CallFunction(reinterpret_cast<PyObject*>(baton), const_cast<char*>("s"), str);
      SWIG_PYTHON_THREAD_END_BLOCK;
    }
}
%}
