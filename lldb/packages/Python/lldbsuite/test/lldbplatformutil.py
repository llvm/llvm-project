""" This module contains functions used by the test cases to hide the
architecture and/or the platform dependent nature of the tests. """

from __future__ import absolute_import

# System modules

# Third-party modules

# LLDB modules

import re

def check_first_register_readable(test_case):
    arch = test_case.getArchitecture()

    if arch in ['x86_64', 'i386']:
        test_case.expect("register read eax", substrs = ['eax = 0x'])
    elif arch in ['arm']:
    	test_case.expect("register read r0", substrs = ['r0 = 0x'])
    elif arch in ['aarch64']:
        test_case.expect("register read x0", substrs = ['x0 = 0x'])
    elif re.match("mips",arch):
        test_case.expect("register read zero", substrs = ['zero = 0x'])
    else:
        # TODO: Add check for other architectures
        test_case.fail("Unsupported architecture for test case (arch: %s)" % test_case.getArchitecture())
