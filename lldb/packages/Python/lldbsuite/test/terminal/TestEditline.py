"""
Test that the lldb editline handling is configured correctly.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbpexpect import PExpectTest


class EditlineTest(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    def test_left_right_arrow(self):
        """Test that ctrl+left/right arrow navigates words correctly.

        Note: just sending escape characters to pexpect and checking the buffer
        doesn't work well, so we run real commands. We want to type
        "help command" while exercising word-navigation, so type it as below,
        where [] indicates cursor position.

        1. Send "el ommand" -> "el ommand[]"
        2. Ctrl+left once   -> "el []ommand"
        3. Send "c"         -> "el c[]ommand"
        4. Ctrl+left twice  -> "[]el command"
        5. Send "h"         -> "h[]el command"
        6. Ctrl+right       -> "hel[] command"
        7. Send "p"         -> "help command"
        """
        self.launch()

        # Run help for different commands for escape variants to make sure each
        # one matches uniquely (the buffer isn't cleared in between matches).
        cases = [
            ("print", "\x1b[1;5D", "\x1b[1;5C"),
            ("step", "\x1b[5D", "\x1b[5C"),
            ("exit", "\x1b\x1b[D", "\x1b\x1b[C"),
        ]
        for (cmd, l_escape, r_escape) in cases:
            self.expect("el {cmd_tail}{L}{cmd_head}{L}{L}h{R}p".format(
                cmd_head=cmd[0], cmd_tail=cmd[1:], L=l_escape, R=r_escape),
                substrs=["Syntax: %s" % cmd])

        self.quit()
