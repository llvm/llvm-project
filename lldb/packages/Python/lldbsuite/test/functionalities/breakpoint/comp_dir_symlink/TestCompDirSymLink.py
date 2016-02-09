"""
Test breakpoint command with AT_comp_dir set to symbolic link.
"""
from __future__ import print_function



import os
import shutil
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


_EXE_NAME = 'CompDirSymLink'  # Must match Makefile
_SRC_FILE = 'main.cpp'
_COMP_DIR_SYM_LINK_PROP = 'plugin.symbol-file.dwarf.comp-dir-symlink-paths'

class CompDirSymLinkTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number(_SRC_FILE, '// Set break point at this line.')
        self.src_path = os.path.join(os.getcwd(), _SRC_FILE)

    @skipIf(hostoslist=["windows"])
    def test_symlink_paths_set(self):
        pwd_symlink = self.create_src_symlink()
        self.doBuild(pwd_symlink)
        self.runCmd("settings set %s %s" % (_COMP_DIR_SYM_LINK_PROP, pwd_symlink))
        lldbutil.run_break_set_by_file_and_line(self, self.src_path, self.line)

    @skipIf(hostoslist=no_match(["linux"]))
    def test_symlink_paths_set_procselfcwd(self):
        pwd_symlink = '/proc/self/cwd'
        self.doBuild(pwd_symlink)
        self.runCmd("settings set %s %s" % (_COMP_DIR_SYM_LINK_PROP, pwd_symlink))
        lldbutil.run_break_set_by_file_and_line(self, self.src_path, self.line)

    @skipIf(hostoslist=["windows"])
    def test_symlink_paths_unset(self):
        pwd_symlink = self.create_src_symlink()
        self.doBuild(pwd_symlink)
        self.runCmd('settings clear ' + _COMP_DIR_SYM_LINK_PROP)
        self.assertRaises(AssertionError, lldbutil.run_break_set_by_file_and_line, self, self.src_path, self.line)

    def create_src_symlink(self):
        pwd_symlink = os.path.join(os.getcwd(), 'pwd_symlink')
        if os.path.exists(pwd_symlink):
          os.unlink(pwd_symlink)
        os.symlink(os.getcwd(), pwd_symlink)
        self.addTearDownHook(lambda: os.remove(pwd_symlink))
        return pwd_symlink

    def doBuild(self, pwd_symlink):
        self.build(None, None, {'PWD': pwd_symlink}, True)

        exe = os.path.join(os.getcwd(), _EXE_NAME)
        self.runCmd('file ' + exe, CURRENT_EXECUTABLE_SET)
