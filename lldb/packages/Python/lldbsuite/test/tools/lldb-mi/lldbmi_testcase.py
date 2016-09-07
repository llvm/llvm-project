"""
Base class for lldb-mi test cases.
"""

from __future__ import print_function


from lldbsuite.test.lldbtest import *


class MiTestCaseBase(Base):

    mydir = None
    myexe = "a.out"
    mylog = "child.log"

    def getCategories(self):
        return ['lldb-mi']

    @classmethod
    def classCleanup(cls):
        TestBase.RemoveTempFile(cls.myexe)
        TestBase.RemoveTempFile(cls.mylog)

    def setUp(self):
        Base.setUp(self)
        self.buildDefault()
        self.child_prompt = "(gdb)"

    def tearDown(self):
        if self.TraceOn():
            print("\n\nContents of %s:" % self.mylog)
            try:
                print(open(self.mylog, "r").read())
            except IOError:
                pass
        Base.tearDown(self)

    def spawnLldbMi(self, args=None):
        import pexpect
        self.child = pexpect.spawn("%s --interpreter %s" % (
            self.lldbMiExec, args if args else ""))
        self.child.setecho(True)
        self.child.logfile_read = open(self.mylog, "w")
        # wait until lldb-mi has started up and is ready to go
        self.expect(self.child_prompt, exactly=True)

    def runCmd(self, cmd):
        self.child.sendline(cmd)

    def expect(self, pattern, exactly=False, *args, **kwargs):
        if exactly:
            return self.child.expect_exact(pattern, *args, **kwargs)
        return self.child.expect(pattern, *args, **kwargs)
