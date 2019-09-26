import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestObjcModulesModernTypeLookup(TestBase):
  mydir = TestBase.compute_mydir(__file__)

  @skipUnlessDarwin
  # gmodules causes this to crash as we seem to get a NSURL type from the debug information.
  @skipIf(debug_info="gmodules")
  def test(self):
    self.build()
    # Activate modern-type-lookup.
    # FIXME: This has to happen before we create any target otherwise we crash...
    self.runCmd("settings set target.experimental.use-modern-type-lookup true")
    (target, process, thread, main_breakpoint) = lldbutil.run_to_source_breakpoint(self,
          "break here", lldb.SBFileSpec("main.m"))
    self.expect("expr @import Foundation")
    self.expect(
            "p *[NSURL URLWithString:@\"http://lldb.llvm.org\"]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "NSURL",
                "isa",
                "_urlString"])
