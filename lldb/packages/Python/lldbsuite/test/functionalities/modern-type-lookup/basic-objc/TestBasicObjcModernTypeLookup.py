import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBasicObjcModernTypeLookup(TestBase):
  mydir = TestBase.compute_mydir(__file__)

  @skipUnlessDarwin
  def test(self):
    self.build()
    # Activate modern-type-lookup.
    # FIXME: This has to happen before we create any target otherwise we crash...
    self.runCmd("settings set target.experimental.use-modern-type-lookup true")
    (target, process, thread, main_breakpoint) = lldbutil.run_to_source_breakpoint(self,
          "break here", lldb.SBFileSpec("main.m"))
    self.expect("expr 1", substrs=["(int) ", " = 1"])
    self.expect("expr (int)[Foo bar:22]", substrs=["(int) ", " = 44"])
