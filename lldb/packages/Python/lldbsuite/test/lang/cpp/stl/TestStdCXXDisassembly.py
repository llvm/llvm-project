"""
Test the lldb disassemble command on lib stdc++.
"""

from __future__ import print_function


import unittest2
import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class StdCXXDisassembleTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// Set break point at this line.')

    # rdar://problem/8504895
    # Crash while doing 'disassemble -n "-[NSNumber descriptionWithLocale:]"
    @unittest2.skipIf(
        TestBase.skipLongRunningTest(),
        "Skip this long running test")
    def test_stdcxx_disasm(self):
        """Do 'disassemble' on each and every 'Code' symbol entry from the std c++ lib."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # rdar://problem/8543077
        # test/stl: clang built binaries results in the breakpoint locations = 3,
        # is this a problem with clang generated debug info?
        #
        # Break on line 13 of main.cpp.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Now let's get the target as well as the process objects.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # The process should be in a 'stopped' state.
        self.expect(str(process), STOPPED_DUE_TO_BREAKPOINT, exe=False,
                    substrs=["a.out",
                             "stopped"])

        # Disassemble the functions on the call stack.
        self.runCmd("thread backtrace")
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)
        depth = thread.GetNumFrames()
        for i in range(depth - 1):
            frame = thread.GetFrameAtIndex(i)
            function = frame.GetFunction()
            if function.GetName():
                self.runCmd("disassemble -n '%s'" % function.GetName())

        lib_stdcxx = "FAILHORRIBLYHERE"
        # Iterate through the available modules, looking for stdc++ library...
        for i in range(target.GetNumModules()):
            module = target.GetModuleAtIndex(i)
            fs = module.GetFileSpec()
            if (fs.GetFilename().startswith("libstdc++")
                    or fs.GetFilename().startswith("libc++")):
                lib_stdcxx = str(fs)
                break

        # At this point, lib_stdcxx is the full path to the stdc++ library and
        # module is the corresponding SBModule.

        self.expect(lib_stdcxx, "Libraray StdC++ is located", exe=False,
                    substrs=["lib"])

        self.runCmd("image dump symtab '%s'" % lib_stdcxx)
        raw_output = self.res.GetOutput()
        # Now, look for every 'Code' symbol and feed its load address into the
        # command: 'disassemble -s load_address -e end_address', where the
        # end_address is taken from the next consecutive 'Code' symbol entry's
        # load address.
        #
        # The load address column comes after the file address column, with both
        # looks like '0xhhhhhhhh', i.e., 8 hexadecimal digits.
        codeRE = re.compile(r"""
                             \ Code\ {9}      # ' Code' followed by 9 SPCs,
                             0x[0-9a-f]{16}   # the file address column, and
                             \                # a SPC, and
                             (0x[0-9a-f]{16}) # the load address column, and
                             .*               # the rest.
                             """, re.VERBOSE)
        # Maintain a start address variable; if we arrive at a consecutive Code
        # entry, then the load address of the that entry is fed as the end
        # address to the 'disassemble -s SA -e LA' command.
        SA = None
        for line in raw_output.split(os.linesep):
            match = codeRE.search(line)
            if match:
                LA = match.group(1)
                if self.TraceOn():
                    print("line:", line)
                    print("load address:", LA)
                    print("SA:", SA)
                if SA and LA:
                    if int(LA, 16) > int(SA, 16):
                        self.runCmd("disassemble -s %s -e %s" % (SA, LA))
                SA = LA
            else:
                # This entry is not a Code entry.  Reset SA = None.
                SA = None
