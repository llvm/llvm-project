"""
Test that we correctly read the YMM registers.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestYMMRegister(TestBase):
    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIfiOSSimulator
    @skipIf(archs=no_match(['i386', 'x86_64']))
    @debugserver_test
    @skipUnlessFeature('hw.optional.avx512f')
    def test(self):
        self.build(dictionary={"CFLAGS_EXTRAS": "-mavx512f"})
        self.setTearDownCleanup()

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        self.assertTrue(target, VALID_TARGET)

        byte_pattern1 = 0x80
        byte_pattern2 = 0xFF

        # This test is working with assembly instructions; it'll make
        # it easier to debug the console output if the assembly is
        # displayed.
        self.runCmd("settings set stop-disassembly-display always")

        # Launch the process and stop.
        self.expect("run", PROCESS_STOPPED, substrs=['stopped'])

        # Check stop reason; Should be either signal SIGTRAP or EXC_BREAKPOINT
        output = self.res.GetOutput()
        matched = False
        substrs = [
            'stop reason = EXC_BREAKPOINT',
            'stop reason = signal SIGTRAP']
        for str1 in substrs:
            matched = output.find(str1) != -1
            with recording(self, False) as sbuf:
                print("%s sub string: %s" % ('Expecting', str1), file=sbuf)
                print("Matched" if matched else "Not Matched", file=sbuf)
            if matched:
                break
        self.assertTrue(matched, STOPPED_DUE_TO_SIGNAL)

        if self.getArchitecture() == 'x86_64':
            register_range = 16
        else:
            register_range = 8
        for i in range(register_range):
            j = i - ((i / 8) * 8)
            self.runCmd("thread step-inst")

            register_byte = (byte_pattern1 | j)
            pattern = "ymm" + str(i) + " = " + str('{') + (
                str(hex(register_byte)) + ' ') * 31 + str(hex(register_byte)) + str('}')

            self.expect(
                "register read ymm" + str(i),
                substrs=[pattern])

            register_byte = (byte_pattern2 | j)
            pattern = "ymm" + str(i) + " = " + str('{') + (
                str(hex(register_byte)) + ' ') * 31 + str(hex(register_byte)) + str('}')

            self.runCmd("thread step-inst")
            self.expect(
                "register read ymm" + str(i),
                substrs=[pattern])

        self.expect("continue", PROCESS_STOPPED, substrs=['stopped'])

        # Check stop reason; Should be either signal SIGTRAP or EXC_BREAKPOINT
        output = self.res.GetOutput()
        matched = False
        substrs = [
            'stop reason = EXC_BREAKPOINT',
            'stop reason = signal SIGTRAP']
        for str1 in substrs:
            matched = output.find(str1) != -1
            with recording(self, False) as sbuf:
                print("%s sub string: %s" % ('Expecting', str1), file=sbuf)
                print("Matched" if matched else "Not Matched", file=sbuf)
            if matched:
                break
        self.assertTrue(matched, STOPPED_DUE_TO_SIGNAL)

        if self.getArchitecture() == 'x86_64':
            register_range = 32
        else:
            register_range = 8
        for i in range(register_range):
            j = i - ((i / 8) * 8)
            self.runCmd("thread step-inst")
            self.runCmd("thread step-inst")

            register_byte = (byte_pattern2 | j)
            pattern = "zmm" + str(i) + " = " + str('{') + (
                str(hex(register_byte)) + ' ') * 63 + str(hex(register_byte)) + str('}')

            self.expect(
                "register read zmm" + str(i),
                substrs=[pattern])

            register_byte = (byte_pattern2 | j)
            pattern = "zmm" + str(i) + " = " + str('{') + (
                str(hex(register_byte)) + ' ') * 63 + str(hex(register_byte)) + str('}')

            self.runCmd("thread step-inst")
            self.expect(
                "register read zmm" + str(i),
                substrs=[pattern])
