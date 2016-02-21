"""
Test lldb 'image list' on object files across multiple architectures.
This exercises classes like ObjectFileELF and their support for opening
foreign-architecture object files.
"""

from __future__ import print_function



import os.path
import re

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestImageListMultiArchitecture(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_image_list_shows_multiple_architectures(self):
        """Test that image list properly shows the correct architecture for a set of different architecture object files."""
        images = {
            "hello-freebsd-10.0-x86_64-clang-3.3": re.compile(r"x86_64-(\*)?-freebsd10.0(-unknown)? x86_64"),
            "hello-freebsd-10.0-x86_64-gcc-4.7.3": re.compile(r"x86_64-(\*)?-freebsd10.0(-unknown)? x86_64"),
            "hello-netbsd-6.1-x86_64-gcc-4.5.3": re.compile(r"x86_64-(\*)?-netbsd(-unknown)? x86_64"),
            "hello-ubuntu-14.04-x86_64-gcc-4.8.2": re.compile(r"x86_64-(\*)?-linux(-unknown)? x86_64"),
            "hello-ubuntu-14.04-x86_64-clang-3.5pre": re.compile(r"x86_64-(\*)?-linux(-unknown)? x86_64"),
            "hello-unknown-kalimba_arch4-kcc-36": re.compile(r"kalimba4-csr-(unknown|\*)(-unknown)? kalimba"),
            "hello-unknown-kalimba_arch5-kcc-39": re.compile(r"kalimba5-csr-(unknown|\*)(-unknown)? kalimba"),
        }

        for image_name in images:
            file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), "bin", image_name))
            expected_triple_and_arch_regex = images[image_name]

            self.runCmd("file {}".format(file_name))
            self.match("image list -t -A", [expected_triple_and_arch_regex])
        # Revert to the host platform after all of this is done
        self.runCmd("platform select host")
