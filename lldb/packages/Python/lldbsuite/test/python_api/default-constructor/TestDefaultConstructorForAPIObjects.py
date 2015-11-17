"""
Test lldb Python API object's default constructor and make sure it is invalid
after initial construction.

There are also some cases of boundary condition testings sprinkled throughout
the tests where None is passed to SB API which expects (const char *) in the
C++ API counterpart.  Passing None should not crash lldb!

There are three exceptions to the above general rules, though; API objects
SBCommadnReturnObject, SBStream, and SBSymbolContextList, are all valid objects
after default construction.
"""

from __future__ import print_function



import os, time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class APIDefaultConstructorTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBAddress(self):
        obj = lldb.SBAddress()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_address
        sb_address.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBBlock(self):
        obj = lldb.SBBlock()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_block
        sb_block.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBBreakpoint(self):
        obj = lldb.SBBreakpoint()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_breakpoint
        sb_breakpoint.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBBreakpointLocation(self):
        obj = lldb.SBBreakpointLocation()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_breakpointlocation
        sb_breakpointlocation.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBBroadcaster(self):
        obj = lldb.SBBroadcaster()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_broadcaster
        sb_broadcaster.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBCommandReturnObject(self):
        """SBCommandReturnObject object is valid after default construction."""
        obj = lldb.SBCommandReturnObject()
        if self.TraceOn():
            print(obj)
        self.assertTrue(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBCommunication(self):
        obj = lldb.SBCommunication()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_communication
        sb_communication.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBCompileUnit(self):
        obj = lldb.SBCompileUnit()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_compileunit
        sb_compileunit.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBDebugger(self):
        obj = lldb.SBDebugger()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_debugger
        sb_debugger.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    # darwin: This test passes with swig 3.0.2, fails w/3.0.5 other tests fail with 2.0.12 http://llvm.org/pr23488
    def test_SBError(self):
        obj = lldb.SBError()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_error
        sb_error.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBEvent(self):
        obj = lldb.SBEvent()
        # This is just to test that typemap, as defined in lldb.swig, works.
        obj2 = lldb.SBEvent(0, "abc")
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_event
        sb_event.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    def test_SBFileSpec(self):
        obj = lldb.SBFileSpec()
        # This is just to test that FileSpec(None) does not crash.
        obj2 = lldb.SBFileSpec(None, True)
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_filespec
        sb_filespec.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBFrame(self):
        obj = lldb.SBFrame()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_frame
        sb_frame.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBFunction(self):
        obj = lldb.SBFunction()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_function
        sb_function.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBInstruction(self):
        obj = lldb.SBInstruction()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_instruction
        sb_instruction.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBInstructionList(self):
        obj = lldb.SBInstructionList()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_instructionlist
        sb_instructionlist.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBLineEntry(self):
        obj = lldb.SBLineEntry()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_lineentry
        sb_lineentry.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBListener(self):
        obj = lldb.SBListener()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_listener
        sb_listener.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    # Py3 asserts due to a bug in SWIG.  Trying to upstream a patch to fix this in 3.0.8
    @skipIf(py_version=['>=', (3,0)], swig_version=['<', (3,0,8)])
    def test_SBModule(self):
        obj = lldb.SBModule()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_module
        sb_module.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBProcess(self):
        obj = lldb.SBProcess()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_process
        sb_process.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBSection(self):
        obj = lldb.SBSection()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_section
        sb_section.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBStream(self):
        """SBStream object is valid after default construction."""
        obj = lldb.SBStream()
        if self.TraceOn():
            print(obj)
        self.assertTrue(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBStringList(self):
        obj = lldb.SBStringList()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_stringlist
        sb_stringlist.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBSymbol(self):
        obj = lldb.SBSymbol()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_symbol
        sb_symbol.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBSymbolContext(self):
        obj = lldb.SBSymbolContext()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_symbolcontext
        sb_symbolcontext.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBSymbolContextList(self):
        """SBSymbolContextList object is valid after default construction."""
        obj = lldb.SBSymbolContextList()
        if self.TraceOn():
            print(obj)
        self.assertTrue(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBTarget(self):
        obj = lldb.SBTarget()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_target
        sb_target.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBThread(self):
        obj = lldb.SBThread()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_thread
        sb_thread.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBType(self):
        try:
            obj = lldb.SBType()
            if self.TraceOn():
                print(obj)
            self.assertFalse(obj)
            # If we reach here, the test fails.
            self.fail("lldb.SBType() should fail, not succeed!")
        except:
            # Exception is expected.
            return
            
        # Unreachable code because lldb.SBType() should fail.
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_type
        sb_type.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBTypeList(self):
        """SBTypeList object is valid after default construction."""
        obj = lldb.SBTypeList()
        if self.TraceOn():
            print(obj)
        self.assertTrue(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBValue(self):
        obj = lldb.SBValue()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_value
        sb_value.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBValueList(self):
        obj = lldb.SBValueList()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_valuelist
        sb_valuelist.fuzz_obj(obj)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_SBWatchpoint(self):
        obj = lldb.SBWatchpoint()
        if self.TraceOn():
            print(obj)
        self.assertFalse(obj)
        # Do fuzz testing on the invalid obj, it should not crash lldb.
        import sb_watchpoint
        sb_watchpoint.fuzz_obj(obj)
