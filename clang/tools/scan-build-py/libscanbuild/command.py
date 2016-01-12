# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
""" This module is responsible for to parse a compiler invocation. """

import re
import os

__all__ = ['Action', 'classify_parameters', 'classify_source']


class Action(object):
    """ Enumeration class for compiler action. """

    Link, Compile, Ignored = range(3)


def classify_parameters(command):
    """ Parses the command line arguments of the given invocation. """

    # result value of this method.
    # some value are preset, some will be set only when found.
    result = {
        'action': Action.Link,
        'files': [],
        'output': None,
        'compile_options': [],
        'c++': is_cplusplus_compiler(command[0])
        # archs_seen
        # language
    }

    # data structure to ignore compiler parameters.
    # key: parameter name, value: number of parameters to ignore afterwards.
    ignored = {
        '-g': 0,
        '-fsyntax-only': 0,
        '-save-temps': 0,
        '-install_name': 1,
        '-exported_symbols_list': 1,
        '-current_version': 1,
        '-compatibility_version': 1,
        '-init': 1,
        '-e': 1,
        '-seg1addr': 1,
        '-bundle_loader': 1,
        '-multiply_defined': 1,
        '-sectorder': 3,
        '--param': 1,
        '--serialize-diagnostics': 1
    }

    args = iter(command[1:])
    for arg in args:
        # compiler action parameters are the most important ones...
        if arg in {'-E', '-S', '-cc1', '-M', '-MM', '-###'}:
            result.update({'action': Action.Ignored})
        elif arg == '-c':
            result.update({'action': max(result['action'], Action.Compile)})
        # arch flags are taken...
        elif arg == '-arch':
            archs = result.get('archs_seen', [])
            result.update({'archs_seen': archs + [next(args)]})
        # explicit language option taken...
        elif arg == '-x':
            result.update({'language': next(args)})
        # output flag taken...
        elif arg == '-o':
            result.update({'output': next(args)})
        # warning disable options are taken...
        elif re.match(r'^-Wno-', arg):
            result['compile_options'].append(arg)
        # warning options are ignored...
        elif re.match(r'^-[mW].+', arg):
            pass
        # some preprocessor parameters are ignored...
        elif arg in {'-MD', '-MMD', '-MG', '-MP'}:
            pass
        elif arg in {'-MF', '-MT', '-MQ'}:
            next(args)
        # linker options are ignored...
        elif arg in {'-static', '-shared', '-s', '-rdynamic'} or \
                re.match(r'^-[lL].+', arg):
            pass
        elif arg in {'-l', '-L', '-u', '-z', '-T', '-Xlinker'}:
            next(args)
        # some other options are ignored...
        elif arg in ignored.keys():
            for _ in range(ignored[arg]):
                next(args)
        # parameters which looks source file are taken...
        elif re.match(r'^[^-].+', arg) and classify_source(arg):
            result['files'].append(arg)
        # and consider everything else as compile option.
        else:
            result['compile_options'].append(arg)

    return result


def classify_source(filename, cplusplus=False):
    """ Return the language from file name extension. """

    mapping = {
        '.c': 'c++' if cplusplus else 'c',
        '.i': 'c++-cpp-output' if cplusplus else 'c-cpp-output',
        '.ii': 'c++-cpp-output',
        '.m': 'objective-c',
        '.mi': 'objective-c-cpp-output',
        '.mm': 'objective-c++',
        '.mii': 'objective-c++-cpp-output',
        '.C': 'c++',
        '.cc': 'c++',
        '.CC': 'c++',
        '.cp': 'c++',
        '.cpp': 'c++',
        '.cxx': 'c++',
        '.c++': 'c++',
        '.C++': 'c++',
        '.txx': 'c++'
    }

    __, extension = os.path.splitext(os.path.basename(filename))
    return mapping.get(extension)


def is_cplusplus_compiler(name):
    """ Returns true when the compiler name refer to a C++ compiler. """

    match = re.match(r'^([^/]*/)*(\w*-)*(\w+\+\+)(-(\d+(\.\d+){0,3}))?$', name)
    return False if match is None else True
