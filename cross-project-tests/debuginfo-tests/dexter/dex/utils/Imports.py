import importlib.util
import os
import sys


def load_module(name, path, mod_file="__init__.py"):
    # The module is either defined by a directory, in which case we search for
    # `path/name/__init__.py`, or it is a single file at `path/mod_file`.
    mod_path = (
        os.path.join(path, name, mod_file)
        if mod_file == "__init__.py"
        else os.path.join(path, mod_file)
    )
    spec = importlib.util.spec_from_file_location(name, mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
