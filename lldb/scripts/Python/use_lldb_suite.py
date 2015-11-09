import inspect
import os
import sys

def find_lldb_root():
    lldb_root = os.path.dirname(inspect.getfile(inspect.currentframe()))
    while True:
        lldb_root = os.path.dirname(lldb_root)
        if lldb_root is None:
            return None

        test_path = os.path.join(lldb_root, "lldb.root")
        if os.path.isfile(test_path):
            return lldb_root
    return None

lldb_root = find_lldb_root()
if lldb_root is not None:
    import imp
    module = imp.find_module("use_lldb_suite_root", [lldb_root])
    if module is not None:
        imp.load_module("use_lldb_suite_root", *module)
