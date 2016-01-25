"""
Test SBProcess APIs, including ReadMemory(), WriteMemory(), and others.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbutil import get_stopped_thread, state_type_to_str
from lldbsuite.test.lldbtest import *

class ProcessAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.cpp", "// Set break point at this line and check variable 'my_char'.")

    @add_test_categories(['pyapi'])
    def test_read_memory(self):
        """Test Python SBProcess.ReadMemory() API."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint")
        frame = thread.GetFrameAtIndex(0)

        # Get the SBValue for the global variable 'my_char'.
        val = frame.FindValue("my_char", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)

        # Due to the typemap magic (see lldb.swig), we pass in 1 to ReadMemory and
        # expect to get a Python string as the result object!
        error = lldb.SBError()
        self.assertFalse(val.TypeIsPointerType())
        content = process.ReadMemory(val.AddressOf().GetValueAsUnsigned(), 1, error)
        if not error.Success():
            self.fail("SBProcess.ReadMemory() failed")
        if self.TraceOn():
            print("memory content:", content)

        self.expect(content, "Result from SBProcess.ReadMemory() matches our expected output: 'x'",
                    exe=False,
            startstr = b'x')

        # Read (char *)my_char_ptr.
        val = frame.FindValue("my_char_ptr", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)
        cstring = process.ReadCStringFromMemory(val.GetValueAsUnsigned(), 256, error)
        if not error.Success():
            self.fail("SBProcess.ReadCStringFromMemory() failed")
        if self.TraceOn():
            print("cstring read is:", cstring)

        self.expect(cstring, "Result from SBProcess.ReadCStringFromMemory() matches our expected output",
                    exe=False,
            startstr = 'Does it work?')

        # Get the SBValue for the global variable 'my_cstring'.
        val = frame.FindValue("my_cstring", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)

        # Due to the typemap magic (see lldb.swig), we pass in 256 to read at most 256 bytes
        # from the address, and expect to get a Python string as the result object!
        self.assertFalse(val.TypeIsPointerType())
        cstring = process.ReadCStringFromMemory(val.AddressOf().GetValueAsUnsigned(), 256, error)
        if not error.Success():
            self.fail("SBProcess.ReadCStringFromMemory() failed")
        if self.TraceOn():
            print("cstring read is:", cstring)

        self.expect(cstring, "Result from SBProcess.ReadCStringFromMemory() matches our expected output",
                    exe=False,
            startstr = 'lldb.SBProcess.ReadCStringFromMemory() works!')

        # Get the SBValue for the global variable 'my_uint32'.
        val = frame.FindValue("my_uint32", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)

        # Due to the typemap magic (see lldb.swig), we pass in 4 to read 4 bytes
        # from the address, and expect to get an int as the result!
        self.assertFalse(val.TypeIsPointerType())
        my_uint32 = process.ReadUnsignedFromMemory(val.AddressOf().GetValueAsUnsigned(), 4, error)
        if not error.Success():
            self.fail("SBProcess.ReadCStringFromMemory() failed")
        if self.TraceOn():
            print("uint32 read is:", my_uint32)

        if my_uint32 != 12345:
            self.fail("Result from SBProcess.ReadUnsignedFromMemory() does not match our expected output")

    @add_test_categories(['pyapi'])
    def test_write_memory(self):
        """Test Python SBProcess.WriteMemory() API."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint")
        frame = thread.GetFrameAtIndex(0)

        # Get the SBValue for the global variable 'my_char'.
        val = frame.FindValue("my_char", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)

        # If the variable does not have a load address, there's no sense continuing.
        if not val.GetLocation().startswith("0x"):
            return

        # OK, let's get the hex location of the variable.
        location = int(val.GetLocation(), 16)

        # The program logic makes the 'my_char' variable to have memory content as 'x'.
        # But we want to use the WriteMemory() API to assign 'a' to the variable.

        # Now use WriteMemory() API to write 'a' into the global variable.
        error = lldb.SBError()
        result = process.WriteMemory(location, 'a', error)
        if not error.Success() or result != 1:
            self.fail("SBProcess.WriteMemory() failed")

        # Read from the memory location.  This time it should be 'a'.
        # Due to the typemap magic (see lldb.swig), we pass in 1 to ReadMemory and
        # expect to get a Python string as the result object!
        content = process.ReadMemory(location, 1, error)
        if not error.Success():
            self.fail("SBProcess.ReadMemory() failed")
        if self.TraceOn():
            print("memory content:", content)

        self.expect(content, "Result from SBProcess.ReadMemory() matches our expected output: 'a'",
                    exe=False,
            startstr = b'a')

    @add_test_categories(['pyapi'])
    def test_access_my_int(self):
        """Test access 'my_int' using Python SBProcess.GetByteOrder() and other APIs."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint")
        frame = thread.GetFrameAtIndex(0)

        # Get the SBValue for the global variable 'my_int'.
        val = frame.FindValue("my_int", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)

        # If the variable does not have a load address, there's no sense continuing.
        if not val.GetLocation().startswith("0x"):
            return

        # OK, let's get the hex location of the variable.
        location = int(val.GetLocation(), 16)

        # Note that the canonical from of the bytearray is little endian.
        from lldbsuite.test.lldbutil import int_to_bytearray, bytearray_to_int

        byteSize = val.GetByteSize()
        bytes = int_to_bytearray(256, byteSize)

        byteOrder = process.GetByteOrder()
        if byteOrder == lldb.eByteOrderBig:
            bytes.reverse()
        elif byteOrder == lldb.eByteOrderLittle:
            pass
        else:
            # Neither big endian nor little endian?  Return for now.
            # Add more logic here if we want to handle other types.
            return

        # The program logic makes the 'my_int' variable to have int type and value of 0.
        # But we want to use the WriteMemory() API to assign 256 to the variable.

        # Now use WriteMemory() API to write 256 into the global variable.
        error = lldb.SBError()
        result = process.WriteMemory(location, bytes, error)
        if not error.Success() or result != byteSize:
            self.fail("SBProcess.WriteMemory() failed")

        # Make sure that the val we got originally updates itself to notice the change:
        self.expect(val.GetValue(),
                    "SBProcess.ReadMemory() successfully writes (int)256 to the memory location for 'my_int'",
                    exe=False,
            startstr = '256')

        # And for grins, get the SBValue for the global variable 'my_int' again, to make sure that also tracks the new value:
        val = frame.FindValue("my_int", lldb.eValueTypeVariableGlobal)
        self.expect(val.GetValue(),
                    "SBProcess.ReadMemory() successfully writes (int)256 to the memory location for 'my_int'",
                    exe=False,
            startstr = '256')

        # Now read the memory content.  The bytearray should have (byte)1 as the second element.
        content = process.ReadMemory(location, byteSize, error)
        if not error.Success():
            self.fail("SBProcess.ReadMemory() failed")

        # The bytearray_to_int utility function expects a little endian bytearray.
        if byteOrder == lldb.eByteOrderBig:
            new_bytes.reverse()

        new_value = bytearray_to_int(content, byteSize)
        if new_value != 256:
            self.fail("Memory content read from 'my_int' does not match (int)256")

        # Dump the memory content....
        if self.TraceOn():
            for i in new_bytes:
                print("byte:", i)

    @add_test_categories(['pyapi'])
    def test_remote_launch(self):
        """Test SBProcess.RemoteLaunch() API with a process not in eStateConnected, and it should fail."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        if self.TraceOn():
            print("process state:", state_type_to_str(process.GetState()))
        self.assertTrue(process.GetState() != lldb.eStateConnected)

        error = lldb.SBError()
        success = process.RemoteLaunch(None, None, None, None, None, None, 0, False, error)
        self.assertTrue(not success, "RemoteLaunch() should fail for process state != eStateConnected")

    @add_test_categories(['pyapi'])
    def test_get_num_supported_hardware_watchpoints(self):
        """Test SBProcess.GetNumSupportedHardwareWatchpoints() API with a process."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        error = lldb.SBError();
        num = process.GetNumSupportedHardwareWatchpoints(error)
        if self.TraceOn() and error.Success():
            print("Number of supported hardware watchpoints: %d" % num)

