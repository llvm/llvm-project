"""
Test some SBModule and SBSection APIs.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbutil import symbol_type_to_str


class ModuleAndSectionAPIsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # Py3 asserts due to a bug in SWIG.  A fix for this was upstreamed into
    # SWIG 3.0.8.
    @skipIf(py_version=['>=', (3, 0)], swig_version=['<', (3, 0, 8)])
    @add_test_categories(['pyapi'])
    def test_module_and_section(self):
        """Test module and section APIs."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.assertTrue(target.GetNumModules() > 0)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print("Number of modules for the target: %d" % target.GetNumModules())
        for module in target.module_iter():
            print(module)

        # Get the executable module at index 0.
        exe_module = target.GetModuleAtIndex(0)

        print("Exe module: %s" % str(exe_module))
        print("Number of sections: %d" % exe_module.GetNumSections())
        print("Number of symbols: %d" % len(exe_module))
        INDENT = ' ' * 4
        INDENT2 = INDENT * 2
        for sec in exe_module.section_iter():
            print(sec)
            print(
                INDENT +
                "Number of subsections: %d" %
                sec.GetNumSubSections())
            if sec.GetNumSubSections() == 0:
                for sym in exe_module.symbol_in_section_iter(sec):
                    print(INDENT + str(sym))
                    print(
                        INDENT +
                        "symbol type: %s" %
                        symbol_type_to_str(
                            sym.GetType()))
            else:
                for subsec in sec:
                    print(INDENT + str(subsec))
                    # Now print the symbols belonging to the subsection....
                    for sym in exe_module.symbol_in_section_iter(subsec):
                        print(INDENT2 + str(sym))
                        print(
                            INDENT2 +
                            "symbol type: %s" %
                            symbol_type_to_str(
                                sym.GetType()))

    @add_test_categories(['pyapi'])
    def test_module_and_section_boundary_condition(self):
        """Test module and section APIs by passing None when it expects a Python string."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.assertTrue(target.GetNumModules() > 0)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print("Number of modules for the target: %d" % target.GetNumModules())
        for module in target.module_iter():
            print(module)

        # Get the executable module at index 0.
        exe_module = target.GetModuleAtIndex(0)

        print("Exe module: %s" % str(exe_module))
        print("Number of sections: %d" % exe_module.GetNumSections())

        # Boundary condition testings.  Should not crash lldb!
        exe_module.FindFirstType(None)
        exe_module.FindTypes(None)
        exe_module.FindGlobalVariables(target, None, 1)
        exe_module.FindFunctions(None, 0)
        exe_module.FindSection(None)

        # Get the section at index 1.
        if exe_module.GetNumSections() > 1:
            sec1 = exe_module.GetSectionAtIndex(1)
            print(sec1)
        else:
            sec1 = None

        if sec1:
            sec1.FindSubSection(None)

    @add_test_categories(['pyapi'])
    def test_module_compile_unit_iter(self):
        """Test module's compile unit iterator APIs."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.assertTrue(target.GetNumModules() > 0)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print("Number of modules for the target: %d" % target.GetNumModules())
        for module in target.module_iter():
            print(module)

        # Get the executable module at index 0.
        exe_module = target.GetModuleAtIndex(0)

        print("Exe module: %s" % str(exe_module))
        print("Number of compile units: %d" % exe_module.GetNumCompileUnits())
        INDENT = ' ' * 4
        INDENT2 = INDENT * 2
        for cu in exe_module.compile_unit_iter():
            print(cu)

    @add_test_categories(['pyapi'])
    def test_find_compile_units(self):
        """Exercise SBModule.FindCompileUnits() API."""
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.find_compile_units(self.getBuildArtifact('b.out'))

    def find_compile_units(self, exe):
        """Exercise SBModule.FindCompileUnits() API."""
        source_name_list = ["main.cpp", "b.cpp", "c.cpp"]

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        num_modules = target.GetNumModules()
        for i in range(num_modules):
            module = target.GetModuleAtIndex(i)
            for source_name in source_name_list:
                list = module.FindCompileUnits(lldb.SBFileSpec(source_name, False))
                for sc in list:
                    self.assertTrue(
                        sc.GetCompileUnit().GetFileSpec().GetFilename() ==
                        source_name)
