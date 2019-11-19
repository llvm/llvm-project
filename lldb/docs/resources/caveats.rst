Caveats
=======

.. contents::
   :local:

.. _python_caveat:

Python
------

LLDB has a powerful scripting interface which is accessible through Python.
Python is available either from withing LLDB through a (interactive) script
interpreter, or as a Python module which you can import from the Python
interpreter.

To make this possible, LLDB links against the Python shared library. Linking
against Python comes with some constraints to be aware of.

1.  It is not possible to build and link LLDB against a Python 3 library and
    use it from Python 2 and vice versa.

2.  It is not possible to build and link LLDB against one distribution on
    Python and use it through a interpreter coming from another distribution.
    For example, on macOS, if you build and link against Python from
    python.org, you cannot import the lldb module from the Python interpreter
    installed with Homebrew.

3.  To use third party Python packages from inside LLDB, you need to install
    them using a utility (such as ``pip``) from the same Python distribution as
    the one used to build and link LLDB.

The previous considerations are especially important during development, but
apply to binary distributions of LLDB as well. For example, the LLDB that comes
with Xcode links against the Python 3 that's part of Xcode. Therefore you
should always use the Python in Xcode (through ``xcrun python3`` or
``/usr/bin/python3``) to import the lldb module or install packages.
