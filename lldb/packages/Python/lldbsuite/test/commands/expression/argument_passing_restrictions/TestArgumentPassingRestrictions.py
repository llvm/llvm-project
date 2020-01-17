"""
This is a test to ensure that both lldb is reconstructing the right
calling convention for a CXXRecordDecl as represented by:

   DW_CC_pass_by_reference
   DW_CC_pass_by_value

and to also make sure that the ASTImporter is copying over this
setting when importing the CXXRecordDecl via setArgPassingRestrictions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestArgumentPassingRestrictions(TestBase):

  mydir = TestBase.compute_mydir(__file__)

  @skipIf(compiler="clang", compiler_version=['<', '7.0'])
  def test_argument_passing_restrictions(self):
    self.build()

    lldbutil.run_to_source_breakpoint(self, '// break here',
            lldb.SBFileSpec("main.cpp"))

    self.expect("expr returnPassByRef()",
            substrs=['(PassByRef)', '= 11223344'])

    self.expect("expr takePassByRef(p)",
            substrs=['(int)', '= 42'])
