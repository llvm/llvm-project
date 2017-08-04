from __future__ import print_function
from __future__ import absolute_import

import re

import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
import lldbsuite.test.test_categories as test_categories
# System modules
import os
import sys

# Third-party modules

# LLDB modules
import lldb
from .lldbtest import *
from . import configuration
from . import lldbutil
from .decorators import *


def inputFile():
    return "input.swift"


def mainSourceFile():
    return "main.swift"


def breakpointMarker():
    return "Set breakpoint here."


class CommandParser:

    def __init__(self, test):
        self.breakpoint = None
        self.exprs_and_regexps = []
        self.test = test

    def parse_input(self):
        file_handle = open(inputFile(), 'r')
        lines = file_handle.readlines()
        current_expression = None
        for line in lines:
            if line.startswith('///'):
                regexp = line[3:]
                if current_expression:
                    self.exprs_and_regexps.append(
                        {'expr': current_expression, 'regexps': [regexp.strip()]})
                    current_expression = None
                else:
                    if len(self.exprs_and_regexps):
                        self.exprs_and_regexps[-1][
                            'regexps'].append(regexp.strip())
                    else:
                        sys.exit("Failure parsing test: regexp with no command")
            else:
                if current_expression:
                    current_expression += line
                else:
                    current_expression = line

    def set_breakpoint(self, target):
        self.breakpoint = target.BreakpointCreateBySourceRegex(
            breakpointMarker(), lldb.SBFileSpec(mainSourceFile()))

    def handle_breakpoint(self, test, thread, breakpoint_id):
        if self.breakpoint.GetID() == breakpoint_id:
            frame = thread.GetSelectedFrame()
            if test.TraceOn():
                print('Stopped at: %s' % frame)
            options = lldb.SBExpressionOptions()
            options.SetLanguage(lldb.eLanguageTypeSwift)
            options.SetREPLMode(True)
            options.SetFetchDynamicValue(lldb.eDynamicDontRunTarget)

            for expr_and_regexp in self.exprs_and_regexps:
                ret = frame.EvaluateExpression(
                    expr_and_regexp['expr'], options)
                desc_stream = lldb.SBStream()
                ret.GetDescription(desc_stream)
                desc = desc_stream.GetData()
                if test.TraceOn():
                    print("%s --> %s" % (expr_and_regexp['expr'], desc))
                for regexp in expr_and_regexp['regexps']:
                    test.assertTrue(
                        re.search(
                            regexp,
                            desc),
                        "Output of REPL input\n" +
                        expr_and_regexp['expr'] +
                        "was\n" +
                        desc +
                        "which didn't match regexp " +
                        regexp)

            return


class REPLTest(TestBase):
    # Internal implementation

    def getRerunArgs(self):
        # The -N option says to NOT run a if it matches the option argument, so
        # if we are using dSYM we say to NOT run dwarf (-N dwarf) and vice
        # versa.
        if self.using_dsym is None:
            # The test was skipped altogether.
            return ""
        elif self.using_dsym:
            return "-N dwarf %s" % (self.mydir)
        else:
            return "-N dsym %s" % (self.mydir)

    def BuildSourceFile(self):
        if os.path.exists(mainSourceFile()):
            return

        source_file = open(mainSourceFile(), 'w+')
        source_file.write("func stop_here() {\n")
        source_file.write("  // " + breakpointMarker() + "\n")
        source_file.write("}\n")
        source_file.write("stop_here()\n")
        source_file.close()

        return

    def BuildMakefile(self):
        if os.path.exists("Makefile"):
            return

        makefile = open("Makefile", 'w+')

        level = os.sep.join(
            [".."] * len(self.mydir.split(os.sep))) + os.sep + "make"

        makefile.write("LEVEL = " + level + "\n")
        makefile.write("SWIFT_SOURCES := " + mainSourceFile() + "\n")

        makefile.write("include $(LEVEL)/Makefile.rules\n")
        makefile.flush()
        makefile.close()

    @skipUnlessDarwin
    def __test_with_dsym(self):
        return

    def __test_with_dwarf(self):
        self.using_dsym = False
        self.BuildSourceFile()
        self.BuildMakefile()
        self.buildDwarf()
        self.do_test()

    def __test_with_dwo(self):
        return

    def __test_with_gmodules(self):
        return

    def execute_user_command(self, __command):
        exec(__command, globals(), locals())

    def do_test(self):
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)
        target = self.dbg.CreateTarget(exe)

        parser = CommandParser(self)
        parser.parse_input()
        parser.set_breakpoint(target)

        process = target.LaunchSimple(None, None, os.getcwd())

        while lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint):
            thread = lldbutil.get_stopped_thread(
                process, lldb.eStopReasonBreakpoint)
            breakpoint_id = thread.GetStopReasonDataAtIndex(0)
            parser.handle_breakpoint(self, thread, breakpoint_id)
            process.Continue()


def ApplyDecoratorsToFunction(func, decorators):
    tmp = func
    if isinstance(decorators, list):
        for decorator in decorators:
            tmp = decorator(tmp)
    elif hasattr(decorators, '__call__'):
        tmp = decorators(tmp)
    return tmp


def MakeREPLTest(__file, __globals, decorators=None):
    # Adjust the filename if it ends in .pyc.  We want filenames to
    # reflect the source python file, not the compiled variant.
    if __file is not None and __file.endswith(".pyc"):
        # Strip the trailing "c"
        __file = __file[0:-1]

    # Derive the test name from the current file name
    file_basename = os.path.basename(__file)
    REPLTest.mydir = TestBase.compute_mydir(__file)

    test_name, _ = os.path.splitext(file_basename)
    # Build the test case
    test = type(test_name, (REPLTest,), {'using_dsym': None})
    test.name = test_name

    target_platform = lldb.DBG.GetSelectedPlatform().GetTriple().split('-')[2]
    if test_categories.is_supported_on_platform(
            "dwarf", target_platform, configuration.compilers):
        test.test_with_dwarf = ApplyDecoratorsToFunction(
            test._REPLTest__test_with_dwarf, decorators)

    # Add the test case to the globals, and hide REPLTest
    __globals.update({test_name: test})

    # Keep track of the original test filename so we report it
    # correctly in test results.
    test.test_filename = __file
    return test
