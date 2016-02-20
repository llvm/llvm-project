// NOTE: Since Python may define some pre-processor definitions which affect the
// standard headers on some systems, you must include Python.h before any
// standard headers are included.
#include "Python.h"

#include <stdio.h>

#ifndef LLDB_DISABLE_LIBEDIT
#include <editline/readline.h>
#endif

// Simple implementation of the Python readline module using libedit.
// In the event that libedit is excluded from the build, this turns
// back into a null implementation that blocks the module from pulling
// in the GNU readline shared lib, which causes linkage confusion when
// both readline and libedit's readline compatibility symbols collide.
//
// Currently it only installs a PyOS_ReadlineFunctionPointer, without
// implementing any of the readline module methods. This is meant to
// work around LLVM pr18841 to avoid seg faults in the stock Python
// readline.so linked against GNU readline.

static struct PyMethodDef moduleMethods[] =
{
    {nullptr, nullptr, 0, nullptr}
};

#ifndef LLDB_DISABLE_LIBEDIT
PyDoc_STRVAR(
    moduleDocumentation,
    "Simple readline module implementation based on libedit.");
#else
PyDoc_STRVAR(
    moduleDocumentation,
    "Stub module meant to avoid linking GNU readline.");
#endif

#ifndef LLDB_DISABLE_LIBEDIT
static char*
simple_readline(FILE *stdin, FILE *stdout, char *prompt)
{
    rl_instream = stdin;
    rl_outstream = stdout;
    char* line = readline(prompt);
    if (!line)
    {
        char* ret = (char*)PyMem_Malloc(1);
        if (ret != NULL)
            *ret = '\0';
        return ret;
    }
    if (*line)
        add_history(line);
    int n = strlen(line);
    char* ret = (char*)PyMem_Malloc(n + 2);
    strncpy(ret, line, n);
    free(line);
    ret[n] = '\n';
    ret[n+1] = '\0';
    return ret;
}
#endif

PyMODINIT_FUNC
initreadline(void)
{
#ifndef LLDB_DISABLE_LIBEDIT
    PyOS_ReadlineFunctionPointer = simple_readline;
#endif
    Py_InitModule4(
        "readline",
        moduleMethods,
        moduleDocumentation,
        static_cast<PyObject *>(NULL),
        PYTHON_API_VERSION);
}
