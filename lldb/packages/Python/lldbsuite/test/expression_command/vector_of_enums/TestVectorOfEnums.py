"""
Test Expression Parser regression test to ensure that we handle enums
correctly, in this case specifically std::vector of enums.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestVectorOfEnums(TestBase):

  mydir = TestBase.compute_mydir(__file__)

  @add_test_categories(["libc++"])
  def test_vector_of_enums(self):
    self.build()

    lldbutil.run_to_source_breakpoint(self, '// break here',
            lldb.SBFileSpec("main.cpp", False))

    self.expect("expr v", substrs=[
         'size=3',
         '[0] = a',
         '[1] = b',
         '[2] = c',
         '}'
        ])
