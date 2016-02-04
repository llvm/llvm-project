from __future__ import print_function
from __future__ import absolute_import

# System modules
from distutils.version import LooseVersion
from functools import wraps
import re
import sys

# Third-party modules
import six
import unittest2

# LLDB modules
import use_lldb_suite

import lldb
from . import configuration
from .result_formatter import EventBuilder
from lldbsuite.support import funcutils
from lldbsuite.test import lldbplatformutil

class DecorateMode:
    Skip, Xfail = range(2)


def _check_expected_version(comparison, expected, actual):
    def fn_leq(x,y): return x <= y
    def fn_less(x,y): return x < y
    def fn_geq(x,y): return x >= y
    def fn_greater(x,y): return x > y
    def fn_eq(x,y): return x == y
    def fn_neq(x,y): return x != y

    op_lookup = {
        "==": fn_eq,
        "=": fn_eq,
        "!=": fn_neq,
        "<>": fn_neq,
        ">": fn_greater,
        "<": fn_less,
        ">=": fn_geq,
        "<=": fn_leq
        }
    expected_str = '.'.join([str(x) for x in expected])
    actual_str = '.'.join([str(x) for x in actual])

    return op_lookup[comparison](LooseVersion(actual_str), LooseVersion(expected_str))

def _check_list_or_lambda(list_or_lambda, value):
    if six.callable(list_or_lambda):
        return list_or_lambda(value)
    elif isinstance(list_or_lambda, list):
        for item in list_or_lambda:
            if value in item:
                return True
        return False
    elif isinstance(list_or_lambda, str):
        return value is None or value in list_or_lambda
    else:
        return list_or_lambda is None or value is None or list_or_lambda == value

def expectedFailure(expected_fn, bugnumber=None):
    def expectedFailure_impl(func):
        if isinstance(func, type) and issubclass(func, unittest2.TestCase):
            raise Exception("Decorator can only be used to decorate a test method")
        @wraps(func)
        def wrapper(*args, **kwargs):
            from unittest2 import case
            self = args[0]
            if funcutils.requires_self(expected_fn):
                xfail_reason = expected_fn(self)
            else:
                xfail_reason = expected_fn()
            if xfail_reason is not None:
                if configuration.results_formatter_object is not None:
                    # Mark this test as expected to fail.
                    configuration.results_formatter_object.handle_event(
                        EventBuilder.event_for_mark_test_expected_failure(self))
                xfail_func = unittest2.expectedFailure(func)
                xfail_func(*args, **kwargs)
            else:
                func(*args, **kwargs)
        return wrapper
    # Some decorators can be called both with no arguments (e.g. @expectedFailureWindows)
    # or with arguments (e.g. @expectedFailureWindows(compilers=['gcc'])).  When called
    # the first way, the first argument will be the actual function because decorators are
    # weird like that.  So this is basically a check that says "which syntax was the original
    # function decorated with?"
    if six.callable(bugnumber):
        return expectedFailure_impl(bugnumber)
    else:
        return expectedFailure_impl

def skipTestIfFn(expected_fn, bugnumber=None):
    def skipTestIfFn_impl(func):
        if isinstance(func, type) and issubclass(func, unittest2.TestCase):
            raise Exception("@skipTestIfFn can only be used to decorate a test method")

        @wraps(func)
        def wrapper(*args, **kwargs):
            from unittest2 import case
            self = args[0]
            if funcutils.requires_self(expected_fn):
                reason = expected_fn(self)
            else:
                reason = expected_fn()

            if reason is not None:
               self.skipTest(reason)
            else:
                func(*args, **kwargs)
        return wrapper

    # Some decorators can be called both with no arguments (e.g. @expectedFailureWindows)
    # or with arguments (e.g. @expectedFailureWindows(compilers=['gcc'])).  When called
    # the first way, the first argument will be the actual function because decorators are
    # weird like that.  So this is basically a check that says "how was the decorator used"
    if six.callable(bugnumber):
        return skipTestIfFn_impl(bugnumber)
    else:
        return skipTestIfFn_impl

def _decorateTest(mode,
                 bugnumber=None, oslist=None, hostoslist=None,
                 compiler=None, compiler_version=None,
                 archs=None, triple=None,
                 debug_info=None,
                 swig_version=None, py_version=None,
                 remote=None):
    def fn(self):
        skip_for_os = _check_list_or_lambda(oslist, self.getPlatform())
        skip_for_hostos = _check_list_or_lambda(hostoslist, lldbplatformutil.getHostPlatform())
        skip_for_compiler = _check_list_or_lambda(self.getCompiler(), compiler) and self.expectedCompilerVersion(compiler_version)
        skip_for_arch = _check_list_or_lambda(archs, self.getArchitecture())
        skip_for_debug_info = _check_list_or_lambda(debug_info, self.debug_info)
        skip_for_triple = triple is None or re.match(triple, lldb.DBG.GetSelectedPlatform().GetTriple())
        skip_for_swig_version = (swig_version is None) or (not hasattr(lldb, 'swig_version')) or (_check_expected_version(swig_version[0], swig_version[1], lldb.swig_version))
        skip_for_py_version = (py_version is None) or _check_expected_version(py_version[0], py_version[1], sys.version_info)
        skip_for_remote = (remote is None) or (remote == (lldb.remote_platform is not None))

        # For the test to be skipped, all specified (e.g. not None) parameters must be True.
        # An unspecified parameter means "any", so those are marked skip by default.  And we skip
        # the final test if all conditions are True.
        conditions = [(oslist, skip_for_os, "target o/s"),
                      (hostoslist, skip_for_hostos, "host o/s"),
                      (compiler, skip_for_compiler, "compiler or version"),
                      (archs, skip_for_arch, "architecture"),
                      (debug_info, skip_for_debug_info, "debug info format"),
                      (triple, skip_for_triple, "target triple"),
                      (swig_version, skip_for_swig_version, "swig version"),
                      (py_version, skip_for_py_version, "python version"),
                      (remote, skip_for_remote, "platform locality (remote/local)")]
        reasons = []
        final_skip_result = True
        for this_condition in conditions:
            final_skip_result = final_skip_result and this_condition[1]
            if this_condition[0] is not None and this_condition[1]:
                reasons.append(this_condition[2])
        reason_str = None
        if final_skip_result:
            mode_str = {DecorateMode.Skip : "skipping", DecorateMode.Xfail : "xfailing"}[mode]
            if len(reasons) > 0:
                reason_str = ",".join(reasons)
                reason_str = "{} due to the following parameter(s): {}".format(mode_str, reason_str)
            else:
                reason_str = "{} unconditionally"
        return reason_str

    if mode == DecorateMode.Skip:
        return skipTestIfFn(fn, bugnumber)
    elif mode == DecorateMode.Xfail:
        return expectedFailure(fn, bugnumber)
    else:
        return None

# provide a function to xfail on defined oslist, compiler version, and archs
# if none is specified for any argument, that argument won't be checked and thus means for all
# for example,
# @expectedFailureAll, xfail for all platform/compiler/arch,
# @expectedFailureAll(compiler='gcc'), xfail for gcc on all platform/architecture
# @expectedFailureAll(bugnumber, ["linux"], "gcc", ['>=', '4.9'], ['i386']), xfail for gcc>=4.9 on linux with i386
def expectedFailureAll(bugnumber=None,
                       oslist=None, hostoslist=None,
                       compiler=None, compiler_version=None,
                       archs=None, triple=None,
                       debug_info=None,
                       swig_version=None, py_version=None,
                       remote=None):
    return _decorateTest(DecorateMode.Xfail,
                        bugnumber=bugnumber,
                        oslist=oslist, hostoslist=hostoslist,
                        compiler=compiler, compiler_version=compiler_version,
                        archs=archs, triple=triple,
                        debug_info=debug_info,
                        swig_version=swig_version, py_version=py_version,
                        remote=remote)


# provide a function to skip on defined oslist, compiler version, and archs
# if none is specified for any argument, that argument won't be checked and thus means for all
# for example,
# @skipIf, skip for all platform/compiler/arch,
# @skipIf(compiler='gcc'), skip for gcc on all platform/architecture
# @skipIf(bugnumber, ["linux"], "gcc", ['>=', '4.9'], ['i386']), skip for gcc>=4.9 on linux with i386
def skipIf(bugnumber=None,
           oslist=None, hostoslist=None,
           compiler=None, compiler_version=None,
           archs=None, triple=None,
           debug_info=None,
           swig_version=None, py_version=None,
           remote=None):
    return _decorateTest(DecorateMode.Skip,
                        bugnumber=bugnumber,
                        oslist=oslist, hostoslist=hostoslist,
                        compiler=compiler, compiler_version=compiler_version,
                        archs=archs, triple=triple,
                        debug_info=debug_info,
                        swig_version=swig_version, py_version=py_version,
                        remote=remote)
