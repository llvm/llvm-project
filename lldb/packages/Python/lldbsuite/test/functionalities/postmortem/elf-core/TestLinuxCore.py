"""
Test basics of linux core file debugging.
"""

from __future__ import print_function

import shutil
import struct

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LinuxCoreTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    _i386_pid = 32306
    _x86_64_pid = 32259
    _s390x_pid = 1045
    _mips64_n64_pid = 25619
    _mips64_n32_pid = 3670
    _mips_o32_pid = 3532
    _ppc64le_pid = 28147

    _i386_regions = 4
    _x86_64_regions = 5
    _s390x_regions = 2
    _mips_regions = 5
    _ppc64le_regions = 2

    def setUp(self):
        super(LinuxCoreTestCase, self).setUp()
        self._initial_platform = lldb.DBG.GetSelectedPlatform()

    def tearDown(self):
        lldb.DBG.SetSelectedPlatform(self._initial_platform)
        super(LinuxCoreTestCase, self).tearDown()

    @expectedFailureAll(bugnumber="llvm.org/pr37371", hostoslist=["windows"])
    @skipIf(triple='^mips')
    def test_i386(self):
        """Test that lldb can read the process information from an i386 linux core file."""
        self.do_test("linux-i386", self._i386_pid, self._i386_regions, "a.out")

    @expectedFailureAll(bugnumber="llvm.org/pr37371", hostoslist=["windows"])
    def test_mips_o32(self):
        """Test that lldb can read the process information from an MIPS O32 linux core file."""
        self.do_test("linux-mipsel-gnuabio32", self._mips_o32_pid,
                self._mips_regions, "linux-mipsel-gn")

    @expectedFailureAll(bugnumber="llvm.org/pr37371", hostoslist=["windows"])
    def test_mips_n32(self):
        """Test that lldb can read the process information from an MIPS N32 linux core file """
        self.do_test("linux-mips64el-gnuabin32", self._mips64_n32_pid,
                self._mips_regions, "linux-mips64el-")

    @expectedFailureAll(bugnumber="llvm.org/pr37371", hostoslist=["windows"])
    def test_mips_n64(self):
        """Test that lldb can read the process information from an MIPS N64 linux core file """
        self.do_test("linux-mips64el-gnuabi64", self._mips64_n64_pid,
                self._mips_regions, "linux-mips64el-")

    @expectedFailureAll(bugnumber="llvm.org/pr37371", hostoslist=["windows"])
    @skipIf(triple='^mips')
    def test_ppc64le(self):
        """Test that lldb can read the process information from an ppc64le linux core file."""
        self.do_test("linux-ppc64le", self._ppc64le_pid, self._ppc64le_regions,
                "linux-ppc64le.ou")

    @expectedFailureAll(bugnumber="llvm.org/pr37371", hostoslist=["windows"])
    @skipIf(triple='^mips')
    def test_x86_64(self):
        """Test that lldb can read the process information from an x86_64 linux core file."""
        self.do_test("linux-x86_64", self._x86_64_pid, self._x86_64_regions,
        "a.out")

    @expectedFailureAll(bugnumber="llvm.org/pr37371", hostoslist=["windows"])
    @skipIf(triple='^mips')
    def test_s390x(self):
        """Test that lldb can read the process information from an s390x linux core file."""
        self.do_test("linux-s390x", self._s390x_pid, self._s390x_regions,
        "a.out")

    @expectedFailureAll(bugnumber="llvm.org/pr37371", hostoslist=["windows"])
    @skipIf(triple='^mips')
    def test_same_pid_running(self):
        """Test that we read the information from the core correctly even if we have a running
        process with the same PID around"""
        exe_file = self.getBuildArtifact("linux-x86_64-pid.out")
        core_file = self.getBuildArtifact("linux-x86_64-pid.core")
        shutil.copyfile("linux-x86_64.out", exe_file)
        shutil.copyfile("linux-x86_64.core", core_file)
        with open(core_file, "r+b") as f:
            # These are offsets into the NT_PRSTATUS and NT_PRPSINFO structures in the note
            # segment of the core file. If you update the file, these offsets may need updating
            # as well. (Notes can be viewed with readelf --notes.)
            for pid_offset in [0x1c4, 0x320]:
                f.seek(pid_offset)
                self.assertEqual(
                    struct.unpack(
                        "<I",
                        f.read(4))[0],
                    self._x86_64_pid)

                # We insert our own pid, and make sure the test still
                # works.
                f.seek(pid_offset)
                f.write(struct.pack("<I", os.getpid()))
        self.do_test(self.getBuildArtifact("linux-x86_64-pid"), os.getpid(),
                self._x86_64_regions, "a.out")

    @expectedFailureAll(bugnumber="llvm.org/pr37371", hostoslist=["windows"])
    @skipIf(triple='^mips')
    def test_two_cores_same_pid(self):
        """Test that we handle the situation if we have two core files with the same PID
        around"""
        alttarget = self.dbg.CreateTarget("altmain.out")
        altprocess = alttarget.LoadCore("altmain.core")
        self.assertTrue(altprocess, PROCESS_IS_VALID)
        self.assertEqual(altprocess.GetNumThreads(), 1)
        self.assertEqual(altprocess.GetProcessID(), self._x86_64_pid)

        altframe = altprocess.GetSelectedThread().GetFrameAtIndex(0)
        self.assertEqual(altframe.GetFunctionName(), "_start")
        self.assertEqual(
            altframe.GetLineEntry().GetLine(),
            line_number(
                "altmain.c",
                "Frame _start"))

        error = lldb.SBError()
        F = altprocess.ReadCStringFromMemory(
            altframe.FindVariable("F").GetValueAsUnsigned(), 256, error)
        self.assertTrue(error.Success())
        self.assertEqual(F, "_start")

        # without destroying this process, run the test which opens another core file with the
        # same pid
        self.do_test("linux-x86_64", self._x86_64_pid, self._x86_64_regions,
                "a.out")

    @expectedFailureAll(bugnumber="llvm.org/pr37371", hostoslist=["windows"])
    @skipIf(triple='^mips')
    def test_FPR_SSE(self):
        # check x86_64 core file
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-fpr_sse_x86_64.core")

        values = {}
        values["fctrl"] = "0x037f"
        values["fstat"] = "0x0000"
        values["ftag"] = "0x00ff"
        values["fop"] = "0x0000"
        values["fiseg"] = "0x00000000"
        values["fioff"] = "0x0040011e"
        values["foseg"] = "0x00000000"
        values["fooff"] = "0x00000000"
        values["mxcsr"] = "0x00001f80"
        values["mxcsrmask"] = "0x0000ffff"
        values["st0"] = "{0x99 0xf7 0xcf 0xfb 0x84 0x9a 0x20 0x9a 0xfd 0x3f}"
        values["st1"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x80 0xff 0x3f}"
        values["st2"] = "{0xfe 0x8a 0x1b 0xcd 0x4b 0x78 0x9a 0xd4 0x00 0x40}"
        values["st3"] = "{0xac 0x79 0xcf 0xd1 0xf7 0x17 0x72 0xb1 0xfe 0x3f}"
        values["st4"] = "{0xbc 0xf0 0x17 0x5c 0x29 0x3b 0xaa 0xb8 0xff 0x3f}"
        values["st5"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x80 0xff 0x3f}"
        values["st6"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["st7"] = "{0x35 0xc2 0x68 0x21 0xa2 0xda 0x0f 0xc9 0x00 0x40}"
        values["xmm0"] = "{0x29 0x31 0x64 0x46 0x29 0x31 0x64 0x46 0x29 0x31 0x64 0x46 0x29 0x31 0x64 0x46}"
        values["xmm1"] = "{0x9c 0xed 0x86 0x64 0x9c 0xed 0x86 0x64 0x9c 0xed 0x86 0x64 0x9c 0xed 0x86 0x64}"
        values["xmm2"] = "{0x07 0xc2 0x1f 0xd7 0x07 0xc2 0x1f 0xd7 0x07 0xc2 0x1f 0xd7 0x07 0xc2 0x1f 0xd7}"
        values["xmm3"] = "{0xa2 0x20 0x48 0x25 0xa2 0x20 0x48 0x25 0xa2 0x20 0x48 0x25 0xa2 0x20 0x48 0x25}"
        values["xmm4"] = "{0xeb 0x5a 0xa8 0xc4 0xeb 0x5a 0xa8 0xc4 0xeb 0x5a 0xa8 0xc4 0xeb 0x5a 0xa8 0xc4}"
        values["xmm5"] = "{0x49 0x41 0x20 0x0b 0x49 0x41 0x20 0x0b 0x49 0x41 0x20 0x0b 0x49 0x41 0x20 0x0b}"
        values["xmm6"] = "{0xf8 0xf1 0x8b 0x4f 0xf8 0xf1 0x8b 0x4f 0xf8 0xf1 0x8b 0x4f 0xf8 0xf1 0x8b 0x4f}"
        values["xmm7"] = "{0x13 0xf1 0x30 0xcd 0x13 0xf1 0x30 0xcd 0x13 0xf1 0x30 0xcd 0x13 0xf1 0x30 0xcd}"

        for regname, value in values.iteritems():
            self.expect("register read {}".format(regname), substrs=["{} = {}".format(regname, value)])


        # now check i386 core file
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-fpr_sse_i386.core")

        values["fioff"] = "0x080480cc"

        for regname, value in values.iteritems():
            self.expect("register read {}".format(regname), substrs=["{} = {}".format(regname, value)])

    def check_memory_regions(self, process, region_count):
        region_list = process.GetMemoryRegions()
        self.assertEqual(region_list.GetSize(), region_count)

        region = lldb.SBMemoryRegionInfo()

        # Check we have the right number of regions.
        self.assertEqual(region_list.GetSize(), region_count)

        # Check that getting a region beyond the last in the list fails.
        self.assertFalse(
            region_list.GetMemoryRegionAtIndex(
                region_count, region))

        # Check each region is valid.
        for i in range(region_list.GetSize()):
            # Check we can actually get this region.
            self.assertTrue(region_list.GetMemoryRegionAtIndex(i, region))

            # Every region in the list should be mapped.
            self.assertTrue(region.IsMapped())

            # Test the address at the start of a region returns it's enclosing
            # region.
            begin_address = region.GetRegionBase()
            region_at_begin = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(begin_address, region_at_begin)
            self.assertEqual(region, region_at_begin)

            # Test an address in the middle of a region returns it's enclosing
            # region.
            middle_address = (region.GetRegionBase() +
                              region.GetRegionEnd()) / 2
            region_at_middle = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(
                middle_address, region_at_middle)
            self.assertEqual(region, region_at_middle)

            # Test the address at the end of a region returns it's enclosing
            # region.
            end_address = region.GetRegionEnd() - 1
            region_at_end = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(end_address, region_at_end)
            self.assertEqual(region, region_at_end)

            # Check that quering the end address does not return this region but
            # the next one.
            next_region = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(
                region.GetRegionEnd(), next_region)
            self.assertNotEqual(region, next_region)
            self.assertEqual(
                region.GetRegionEnd(),
                next_region.GetRegionBase())

        # Check that query beyond the last region returns an unmapped region
        # that ends at LLDB_INVALID_ADDRESS
        last_region = lldb.SBMemoryRegionInfo()
        region_list.GetMemoryRegionAtIndex(region_count - 1, last_region)
        end_region = lldb.SBMemoryRegionInfo()
        error = process.GetMemoryRegionInfo(
            last_region.GetRegionEnd(), end_region)
        self.assertFalse(end_region.IsMapped())
        self.assertEqual(
            last_region.GetRegionEnd(),
            end_region.GetRegionBase())
        self.assertEqual(end_region.GetRegionEnd(), lldb.LLDB_INVALID_ADDRESS)

    def check_state(self, process):
        with open(os.devnull) as devnul:
            # sanitize test output
            self.dbg.SetOutputFileHandle(devnul, False)
            self.dbg.SetErrorFileHandle(devnul, False)

            self.assertTrue(process.is_stopped)

            # Process.Continue
            error = process.Continue()
            self.assertFalse(error.Success())
            self.assertTrue(process.is_stopped)

            # Thread.StepOut
            thread = process.GetSelectedThread()
            thread.StepOut()
            self.assertTrue(process.is_stopped)

            # command line
            self.dbg.HandleCommand('s')
            self.assertTrue(process.is_stopped)
            self.dbg.HandleCommand('c')
            self.assertTrue(process.is_stopped)

            # restore file handles
            self.dbg.SetOutputFileHandle(None, False)
            self.dbg.SetErrorFileHandle(None, False)

    def do_test(self, filename, pid, region_count, thread_name):
        target = self.dbg.CreateTarget(filename + ".out")
        process = target.LoadCore(filename + ".core")
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 1)
        self.assertEqual(process.GetProcessID(), pid)

        self.check_state(process)

        thread = process.GetSelectedThread()
        self.assertTrue(thread)
        self.assertEqual(thread.GetThreadID(), pid)
        self.assertEqual(thread.GetName(), thread_name)
        backtrace = ["bar", "foo", "_start"]
        self.assertEqual(thread.GetNumFrames(), len(backtrace))
        for i in range(len(backtrace)):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame)
            self.assertEqual(frame.GetFunctionName(), backtrace[i])
            self.assertEqual(frame.GetLineEntry().GetLine(),
                             line_number("main.c", "Frame " + backtrace[i]))
            self.assertEqual(
                frame.FindVariable("F").GetValueAsUnsigned(), ord(
                    backtrace[i][0]))

        self.check_memory_regions(process, region_count)

        self.dbg.DeleteTarget(target)
