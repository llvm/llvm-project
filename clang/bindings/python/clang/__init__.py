#===- __init__.py - Clang Python Bindings --------------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

r"""
Clang Library Bindings
======================

This package provides access to the Clang compiler and libraries.

The available modules are:

  cindex

    Bindings for the Clang indexing library.
"""


# Python 3 uses unicode for strings. The bindings, in particular the interaction
# with ctypes, need modifying to handle conversions between unicode and
# c-strings.
import sys 
if sys.version_info[0] != 2: 
    raise Exception("Only Python 2 is supported.")

__all__ = ['cindex']

