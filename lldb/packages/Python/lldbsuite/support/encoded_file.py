"""
                     The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Prepares language bindings for LLDB build process.  Run with --help
to see a description of the supported command line arguments.
"""

# Python modules:
import io

# Third party modules
import six

def _encoded_read(old_read, encoding):
    def impl(size):
        result = old_read(size)
        # If this is Python 2 then we need to convert the resulting `unicode` back
        # into a `str` before returning
        if six.PY2:
            result = result.encode(encoding)
        return result
    return impl

def _encoded_write(old_write, encoding):
    def impl(s):
        # If we were asked to write a `str` (in Py2) or a `bytes` (in Py3) decode it
        # as unicode before attempting to write.
        if isinstance(s, six.binary_type):
            s = s.decode(encoding)
        return old_write(s)
    return impl

'''
Create a Text I/O file object that can be written to with either unicode strings or byte strings
under Python 2 and Python 3, and automatically encodes and decodes as necessary to return the
native string type for the current Python version
'''
def open(file, encoding, mode='r', buffering=-1, errors=None, newline=None, closefd=True):
    wrapped_file = io.open(file, mode=mode, buffering=buffering, encoding=encoding,
                           errors=errors, newline=newline, closefd=closefd)
    new_read = _encoded_read(getattr(wrapped_file, 'read'), encoding)
    new_write = _encoded_write(getattr(wrapped_file, 'write'), encoding)
    setattr(wrapped_file, 'read', new_read)
    setattr(wrapped_file, 'write', new_write)
    return wrapped_file
