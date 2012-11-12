#
# edit-swig-python-wrapper-file.py
#
# This script performs some post-processing editing on the C++ file that
# SWIG generates for python, after running on 'lldb.swig'.   In
# particular, on Apple systems we want to include the Python.h file that
# is used in the /System/Library/Frameworks/Python.framework, but on other
# systems we want to include plain <Python.h>.  So we need to replace:
#
# #include <Python.h>
#
# with:
#
# #if defined (__APPLE__)
# #include <Python/Python.h>
# #else
# #include <Python.h>
# #endif
#
# That's what this python script does.
#

import os, sys

include_python = '#include <Python.h>'
include_python_ifdef = '''#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif
'''

if len (sys.argv) > 1:
    input_dir_name = sys.argv[1]
    full_input_name = input_dir_name + "/LLDBWrapPython.cpp"
else:
    input_dir_name = os.environ["SRCROOT"]
    full_input_name = input_dir_name + "/source/LLDBWrapPython.cpp"
full_output_name = full_input_name + ".edited"

with open(full_input_name, 'r') as f_in:
    with open(full_output_name, 'w') as f_out:
        include_python_found = False
        for line in f_in:
            if not include_python_found:
                if line.startswith(include_python):
                    # Write out the modified lines.
                    f_out.write(include_python_ifdef)
                    include_python_found = True
                    continue

            # Otherwise, copy the line verbatim to the output file.
            f_out.write(line)
