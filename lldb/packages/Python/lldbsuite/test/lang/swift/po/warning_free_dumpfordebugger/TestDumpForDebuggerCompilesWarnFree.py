"""
Test that DumpForDebugger.swift compiles free of warnings
"""

from __future__ import print_function


import os, os.path, time
import subprocess
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import imp

class DumpForDebuggerCompilesWarnFreeTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_dump_for_debugger(self):
        """Test that DumpForDebugger.swift compiles free of warnings"""
        path_to_swift = os.path.abspath(os.path.join(os.path.split(os.path.abspath(__file__))[0], "..", "..", "..", "..", "plugins", "swift.py"))
        swiftcc = imp.load_source('swift', path_to_swift).getSwiftCompiler()
        DumpForDebugger = os.path.join(lldb.SBHostOS.GetLLDBPath(lldb.ePathTypeSupportFileDir).fullpath, "DumpForDebugger.swift")
        self.assertTrue(os.path.exists(DumpForDebugger), "DumpForDebugger.swift couldn't be found")
        try:
            outstr, errstr = system([[swiftcc, '-Xfrontend', '-debugger-support', '-warnings-as-errors', DumpForDebugger, '-o', '/dev/null']])
        except subprocess.CalledProcessError as cpe:
            failmessage = "compilation failed.\ncommand ran: %s\nstdout: %s\nstderr: %s" % \
                (cpe.lldb_extensions['command'], cpe.lldb_extensions['stdout_content'], cpe.lldb_extensions['stderr_content'])
            self.fail(failmessage)

