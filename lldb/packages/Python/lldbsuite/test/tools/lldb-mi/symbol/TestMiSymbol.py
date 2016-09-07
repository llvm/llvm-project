"""
Test lldb-mi -symbol-xxx commands.
"""

from __future__ import print_function


import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiSymbolTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="new failure after r256863")
    def test_lldbmi_symbol_list_lines_file(self):
        """Test that 'lldb-mi --interpreter' works for -symbol-list-lines when file exists."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Get address of main and its line
        self.runCmd("-data-evaluate-expression main")
        self.expect(
            "\^done,value=\"0x[0-9a-f]+ \(a.out`main at main.cpp:[0-9]+\)\"")
        addr = int(self.child.after.split("\"")[1].split(" ")[0], 16)
        line = line_number('main.cpp', '// FUNC_main')

        # Test that -symbol-list-lines works on valid data
        self.runCmd("-symbol-list-lines main.cpp")
        self.expect(
            "\^done,lines=\[\{pc=\"0x0*%x\",line=\"%d\"\}(,\{pc=\"0x[0-9a-f]+\",line=\"\d+\"\})+\]" %
            (addr, line))

        # Test that -symbol-list-lines doesn't include lines from other sources
        # by checking the first and last line, and making sure the other lines
        # are between 30 and 39.
        sline = line_number(
            'symbol_list_lines_inline_test2.cpp',
            '// FUNC_gfunc2')
        eline = line_number(
            'symbol_list_lines_inline_test2.cpp',
            '// END_gfunc2')
        self.runCmd("-symbol-list-lines symbol_list_lines_inline_test2.cpp")
        self.expect(
            "\^done,lines=\[\{pc=\"0x[0-9a-f]+\",line=\"%d\"\}(,\{pc=\"0x[0-9a-f]+\",line=\"3\d\"\})*,\{pc=\"0x[0-9a-f]+\",line=\"%d\"\}(,\{pc=\"0x[0-9a-f]+\",line=\"3\d\"\})*\]" %
            (sline, eline))
        # FIXME: This doesn't work for symbol_list_lines_inline_test.cpp due to clang bug llvm.org/pr24716 (fixed in newer versions of clang)
        ##sline = line_number('symbol_list_lines_inline_test.cpp', '// FUNC_gfunc')
        ##eline = line_number('symbol_list_lines_inline_test.cpp', '// STRUCT_s')
        ##self.runCmd("-symbol-list-lines symbol_list_lines_inline_test.cpp")
        ##self.expect("\^done,lines=\[\{pc=\"0x[0-9a-f]+\",line=\"%d\"\}(,\{pc=\"0x[0-9a-f]+\",line=\"3\d\"\})*,\{pc=\"0x[0-9a-f]+\",line=\"%d\"\}\]" % (sline, eline))

        # Test that -symbol-list-lines works on header files by checking the first
        # and last line, and making sure the other lines are under 29.
        sline = line_number('symbol_list_lines_inline_test.h', '// FUNC_ifunc')
        eline = line_number('symbol_list_lines_inline_test.h', '// FUNC_mfunc')
        self.runCmd("-symbol-list-lines symbol_list_lines_inline_test.h")
        self.expect(
            "\^done,lines=\[\{pc=\"0x[0-9a-f]+\",line=\"%d\"\}(,\{pc=\"0x[0-9a-f]+\",line=\"\d\"\})*(,\{pc=\"0x[0-9a-f]+\",line=\"1\d\"\})*,\{pc=\"0x[0-9a-f]+\",line=\"%d\"\}(,\{pc=\"0x[0-9a-f]+\",line=\"2\d\"\})*\]" %
            (sline, eline))

        # Test that -symbol-list-lines fails when file doesn't exist
        self.runCmd("-symbol-list-lines unknown_file")
        self.expect(
            "\^error,message=\"error: No source filenames matched 'unknown_file'\. \"")

        # Test that -symbol-list-lines fails when file is specified using
        # relative path
        self.runCmd("-symbol-list-lines ./main.cpp")
        self.expect(
            "\^error,message=\"error: No source filenames matched '\./main\.cpp'\. \"")

        # Test that -symbol-list-lines works when file is specified using
        # absolute path
        import os
        path = os.path.join(os.getcwd(), "main.cpp")
        self.runCmd("-symbol-list-lines \"%s\"" % path)
        self.expect(
            "\^done,lines=\[\{pc=\"0x0*%x\",line=\"%d\"\}(,\{pc=\"0x[0-9a-f]+\",line=\"\d+\"\})+\]" %
            (addr, line))

        # Test that -symbol-list-lines fails when file doesn't exist
        self.runCmd("-symbol-list-lines unknown_dir/main.cpp")
        self.expect(
            "\^error,message=\"error: No source filenames matched 'unknown_dir/main\.cpp'\. \"")
