import importlib
import os
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, name, "__init__.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
