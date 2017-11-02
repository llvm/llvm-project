"""
Test lldb Python API for setting output and error file handles 
"""

from __future__ import print_function


import contextlib
import os
import io
import re
import platform
import unittest

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StringIO(io.TextIOBase):

    def __init__(self, buf=''):
        self.buf = buf

    def writable(self):
        return True

    def write(self, s):
        self.buf += s
        return len(s)


class BadIO(io.TextIOBase):

    def writable(self):
        return True

    def write(self, s):
        raise Exception('OH NOE')
    

@contextlib.contextmanager
def replace_stdout(new):
    old = sys.stdout
    sys.stdout = new
    try:
        yield
    finally:
        sys.stdout = old


def handle_command(debugger, cmd, raise_on_fail=True, collect_result=True):

    ret = lldb.SBCommandReturnObject()

    if collect_result:
        interpreter = debugger.GetCommandInterpreter()    
        interpreter.HandleCommand(cmd, ret)
    else:
        debugger.HandleCommand(cmd)
        
    if hasattr(debugger, 'FlushDebuggerOutputHandles'):
        debugger.FlushDebuggerOutputHandles()
        
    if collect_result and raise_on_fail and not ret.Succeeded():
        raise Exception
    
    return ret.GetOutput()


        
class FileHandleTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def comment(self, *args):
        if self.session is not None:
            print(*args, file=self.session)

    def skip_windows(self):
        if platform.system() == 'Windows':
            self.skipTest('windows')
        

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_file_out(self):

        debugger = lldb.SBDebugger.Create()
        try:
            with open('output', 'w') as f:
                debugger.SetOutputFileHandle(f, False)
                handle_command(debugger, 'script print("foobar")')

            with open('output', 'r') as f:
                self.assertEqual(f.read().strip(), "foobar")

        finally:
            self.RemoveTempFile('output')
            lldb.SBDebugger.Destroy(debugger)
            

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_file_error(self):

        debugger = lldb.SBDebugger.Create()
        try:
            with open('output', 'w') as f:
                debugger.SetErrorFileHandle(f, False)
                handle_command(debugger, 'lolwut', raise_on_fail=False, collect_result=False)
                
            with open('output', 'r') as f:
                errors = f.read()
                self.assertTrue(re.search(r'error:.*lolwut', errors))

        finally:
            self.RemoveTempFile('output')
            lldb.SBDebugger.Destroy(debugger)
            

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_string_out(self):

        self.skip_windows()

        io = StringIO()
        debugger = lldb.SBDebugger.Create()
        try:
            debugger.SetOutputFileHandle(io, False)
            handle_command(debugger, 'script print("foobar")')

            self.assertEqual(io.buf.strip(), "foobar")
            
        finally:
            lldb.SBDebugger.Destroy(debugger)
                    

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_string_error(self):

        self.skip_windows()

        io = StringIO()
        debugger = lldb.SBDebugger.Create()
        try:
            debugger.SetErrorFileHandle(io, False)
            handle_command(debugger, 'lolwut', raise_on_fail=False, collect_result=False)

            errors = io.buf
            self.assertTrue(re.search(r'error:.*lolwut', errors))

        finally:
            lldb.SBDebugger.Destroy(debugger)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_replace_stdout(self):

        self.skip_windows()

        io = StringIO()
        debugger = lldb.SBDebugger.Create()
        try:

            with replace_stdout(io):
                handle_command(debugger, 'script print("lol, crash")', collect_result=False)

        finally:
            lldb.SBDebugger.Destroy(debugger)


    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_replace_stdout_with_nonfile(self):

        self.skip_windows()

        io = StringIO()

        with replace_stdout(io):

            class Nothing():
                pass

            debugger = lldb.SBDebugger.Create()
            try:
                with replace_stdout(Nothing):
                    self.assertEqual(sys.stdout, Nothing)
                    handle_command(debugger, 'script print("lol, crash")', collect_result=False)
                    self.assertEqual(sys.stdout, Nothing)
            finally:
                lldb.SBDebugger.Destroy(debugger)

            sys.stdout.write("FOO")

        self.assertEqual(io.buf, "FOO")


    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_stream_error(self):

        self.skip_windows()

        messages = list()

        io = BadIO()
        debugger = lldb.SBDebugger.Create()
        try:
            debugger.SetOutputFileHandle(io, False)
            debugger.SetLoggingCallback(messages.append)
            handle_command(debugger, 'log enable lldb script')
            handle_command(debugger, 'script print "foobar"')
            
        finally:
            lldb.SBDebugger.Destroy(debugger)

        for message in messages:
            self.comment("GOT: " + message.strip())

        self.assertTrue(any('OH NOE' in msg for msg in messages))
        self.assertTrue(any('BadIO' in msg for msg in messages))


