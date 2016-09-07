from __future__ import print_function
from __future__ import absolute_import

# System modules
import inspect

# Third-party modules

# LLDB modules


def requires_self(func):
    func_argc = len(inspect.getargspec(func).args)
    if func_argc == 0 or (
        getattr(
            func,
            'im_self',
            None) is not None) or (
            hasattr(
                func,
                '__self__')):
        return False
    else:
        return True
