"""
Test that the debugger can call a *really* new function.
"""

import os
import lldb
import lldbsuite.test.lldbplatformutil as lldbplatformutil
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

def getOSName(os):
    if os == 'macosx': return 'macOS'
    if os == 'ios': return 'iOS'
    if os == 'tvos': return 'tvOS'
    if os == 'watchos': return 'watchOS'
    return os

class TestAvailability(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @skipIf(oslist=['linux', 'windows'])
    def testAvailability(self):
        os_name = getOSName(lldbplatformutil.getPlatform())
        platform = lldb.DBG.GetSelectedPlatform()
        version = '%d.%d.%d'%(platform.GetOSMajorVersion(),
                              platform.GetOSMinorVersion(),
                              platform.GetOSUpdateVersion())
        program = """
@available(%s %s, *) func f() {}

// ---------------------------------------------------------------------
// Method context.
// ---------------------------------------------------------------------
class C1 {
  func method() {
    print("in method") // break_1
  }
}

C1().method() // break_0

// ---------------------------------------------------------------------
// Generic method context.
// ---------------------------------------------------------------------
class C2 {
  func method<T>(_ t: T) {
    print("in method") // break_2
  }
}

C2().method(0)

// ---------------------------------------------------------------------
// Method in generic class context.
// ---------------------------------------------------------------------
class C3<T> {
  func method() {
    print("in method") // break_3
  }
}

C3<Int>().method()

// ---------------------------------------------------------------------
// Generic method in generic class context.
// ---------------------------------------------------------------------
class C4<U> {
  func method<V>(_ v: V) {
    print("in method") // break_4
  }
}

C4<Int>().method(0)

// ---------------------------------------------------------------------
// Function context.
// ---------------------------------------------------------------------
func f1() {
  print("in function") // break_5
}

f1()

// ---------------------------------------------------------------------
// Generic function context.
// ---------------------------------------------------------------------
func f2<T>(_ t: T) {
  print("in function") // break_6
}

f2(0)

// ---------------------------------------------------------------------
// Top-level context.
// ---------------------------------------------------------------------
print("in top_level") // break_7
"""
        with open(self.getBuildArtifact("main.swift"), 'w') as main:
            main.write(program %(os_name, version))

        self.build()
        source_spec = lldb.SBFileSpec("main.swift")
        (target, process, thread, brk0) = \
            lldbutil.run_to_source_breakpoint(self, "break_0", source_spec)

        # Create breakpoints.
        breakpoints = []
        for i in range(1, 8):
            breakpoints.append(target.BreakpointCreateBySourceRegex(
                'break_%d'%i, lldb.SBFileSpec("main.swift")))
            self.assertTrue(breakpoints[-1] and
                            breakpoints[-1].GetNumLocations() >= 1,
                            VALID_BREAKPOINT)

        for breakpoint in breakpoints:
            threads = lldbutil.continue_to_breakpoint(process, breakpoint)
            self.expect("p f()", "can call")

