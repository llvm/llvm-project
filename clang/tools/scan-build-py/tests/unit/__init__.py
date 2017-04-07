# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

from . import test_libear
from . import test_compilation
from . import test_clang
from . import test_report
from . import test_analyze
from . import test_intercept
from . import test_shell


def load_tests(loader, suite, _):
    suite.addTests(loader.loadTestsFromModule(test_libear))
    suite.addTests(loader.loadTestsFromModule(test_compilation))
    suite.addTests(loader.loadTestsFromModule(test_clang))
    suite.addTests(loader.loadTestsFromModule(test_report))
    suite.addTests(loader.loadTestsFromModule(test_analyze))
    suite.addTests(loader.loadTestsFromModule(test_intercept))
    suite.addTests(loader.loadTestsFromModule(test_shell))
    return suite
