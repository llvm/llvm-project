// RUN: %clang -flto -Xclang -disable-O0-optnone -fuse-ld=lld -Wl,--mllvm=--yk-stackmap-spillreloads-fix -Wl,--mllvm=--yk-insert-stackmaps -Wl,--mllvm=--yk-optnone-after-ir-passes -Wl,--lto-newpm-passes=instcombine -Wl,--mllvm=--yk-shadow-stack %s

// Test case scenario extracted from CPython source code: license at
// https://github.com/python/cpython/blob/main/LICENSE.
//
// Checks whether this program compiles as it previously led to an error in the
// FixStackmapsSpillReloads pass.

#include <stdio.h>

struct PyObject {
	int val;
};
typedef struct PyObject PyObject;

struct PyFrameConstructor {
  PyObject *fc_globals;
  PyObject *fc_builtins;
  PyObject *fc_name;
  PyObject *fc_qualname;
  PyObject *fc_code;
  PyObject *fc_defaults;
  PyObject *fc_kwdefaults;
  PyObject *fc_closure;
};
typedef struct PyFrameConstructor PyFrameConstructor;

struct PyThreadState {
	int val;
};
typedef struct PyThreadState PyThreadState;

PyThreadState PYTS;
PyObject PYO;
PyObject **PYOS;

PyThreadState *_PyThreadState_GET() {
    return &PYTS;
}

PyObject *_PyTuple_FromArray(PyObject *const *defs, int defcount) {
    PYO.val = defcount;
    return &PYO;
}

void PyTuple_SET_ITEM(PyObject *o, int i, PyObject *P) {
}

PyObject *_PyEval_BuiltinsFromGlobals(PyThreadState *tstate, PyObject *globals) {
    return globals;
}

PyObject *PyTuple_New(int kwcount) {
    return &PYO;
}

PyObject **PyMem_Malloc(int kwcount) {
    return PYOS;
}

void PyMem_Free(PyObject **PO) {
  printf("%p", PO);
}

void Py_DECREF(PyObject *defaults) {
  defaults->val++;
}

PyObject *_PyEval_Vector(PyThreadState *PO, PyFrameConstructor *PFC, PyObject *locals, PyObject** allargs, int count, PyObject *kwnames) {
  return locals;
}

/* Legacy API */
PyObject *
PyEval_EvalCodeEx(PyObject *_co, PyObject *globals, PyObject *locals,
                  PyObject *const *args, int argcount,
                  PyObject *const *kws, int kwcount,
                  PyObject *const *defs, int defcount,
                  PyObject *kwdefs, PyObject *closure)
{
    PyThreadState *tstate = _PyThreadState_GET();
    PyObject *res;
    PyObject *defaults = _PyTuple_FromArray(defs, defcount);
    if (defaults == NULL) {
        return NULL;
    }
    PyObject *builtins = _PyEval_BuiltinsFromGlobals(tstate, globals); // borrowed ref
    if (builtins == NULL) {
        Py_DECREF(defaults);
        return NULL;
    }
    if (locals == NULL) {
        locals = globals;
    }
    PyObject *kwnames;
    PyObject *const *allargs;
    PyObject **newargs;
    if (kwcount == 0) {
        allargs = args;
        kwnames = NULL;
    }
    else {
        kwnames = PyTuple_New(kwcount);
        if (kwnames == NULL) {
            res = NULL;
            goto fail;
        }
        newargs = PyMem_Malloc(sizeof(PyObject *)*(kwcount+argcount));
        if (newargs == NULL) {
            res = NULL;
            Py_DECREF(kwnames);
            goto fail;
        }
        for (int i = 0; i < argcount; i++) {
            newargs[i] = args[i];
        }
        for (int i = 0; i < kwcount; i++) {
            Py_DECREF(kws[2*i]);
            PyTuple_SET_ITEM(kwnames, i, kws[2*i]);
            newargs[argcount+i] = kws[2*i+1];
        }
        allargs = newargs;
    }
    PyObject **kwargs = PyMem_Malloc(sizeof(PyObject *)*kwcount);
    if (kwargs == NULL) {
        res = NULL;
        Py_DECREF(kwnames);
        goto fail;
    }
    for (int i = 0; i < kwcount; i++) {
        Py_DECREF(kws[2*i]);
        PyTuple_SET_ITEM(kwnames, i, kws[2*i]);
        kwargs[i] = kws[2*i+1];
    }
    PyFrameConstructor constr = {
        .fc_globals = globals,
        .fc_builtins = builtins,
        .fc_name = _co,
        .fc_qualname = _co,
        .fc_code = _co,
        .fc_defaults = defaults,
        .fc_kwdefaults = kwdefs,
        .fc_closure = closure
    };
    res = _PyEval_Vector(tstate, &constr, locals,
                         (PyObject **)allargs, argcount,
                         kwnames);
    if (kwcount) {
        Py_DECREF(kwnames);
        PyMem_Free(newargs);
    }
fail:
    Py_DECREF(defaults);
    return res;
}

int main() {
	PyObject _co;
	PyObject globals;
	PyObject locals;
	PyObject args;
	PyObject kws;
	PyObject defs;
	PyObject kwdefs;
	PyObject *r = PyEval_EvalCodeEx(&_co, &globals, &locals, (PyObject *const *) &args, 0, (PyObject *const *) &kws, 0, (PyObject *const *) &defs, 5, &kwdefs, 0);
	printf("%p\n", r);
}
