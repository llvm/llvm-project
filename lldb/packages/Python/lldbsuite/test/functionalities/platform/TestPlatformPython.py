"""
Test the lldb platform Python API.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PlatformPythonTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_platform_list(self):
        """Test SBDebugger::GetNumPlatforms() & GetPlatformAtIndex() API"""
        # Verify there's only the host platform present by default.
        self.assertEqual(self.dbg.GetNumPlatforms(), 1)
        host_platform = self.dbg.GetPlatformAtIndex(0)
        self.assertTrue(host_platform.IsValid() and
                        host_platform.GetName() == 'host',
                        'Only the host platform is available')
        # Select another platform and verify that the platform is added to
        # the platform list.
        platform_idx = self.dbg.GetNumAvailablePlatforms() - 1
        if platform_idx < 1:
            self.fail('No platforms other than host are available')
        platform_data = self.dbg.GetAvailablePlatformInfoAtIndex(platform_idx)
        platform_name = platform_data.GetValueForKey('name').GetStringValue(100)
        self.assertNotEqual(platform_name, 'host')
        self.dbg.SetCurrentPlatform(platform_name)
        selected_platform = self.dbg.GetSelectedPlatform()
        self.assertTrue(selected_platform.IsValid())
        self.assertEqual(selected_platform.GetName(), platform_name)
        self.assertEqual(self.dbg.GetNumPlatforms(), 2)
        platform = self.dbg.GetPlatformAtIndex(1)
        self.assertEqual(platform.GetName(), platform_name)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_available_platform_list(self):
        """Test SBDebugger::GetNumAvailablePlatforms() and GetAvailablePlatformInfoAtIndex() API"""
        num_platforms = self.dbg.GetNumAvailablePlatforms()
        self.assertGreater(
            num_platforms, 0,
            'There should be at least one platform available')

        for i in range(num_platforms):
            platform_data = self.dbg.GetAvailablePlatformInfoAtIndex(i)
            name_data = platform_data.GetValueForKey('name')
            desc_data = platform_data.GetValueForKey('description')
            self.assertTrue(
                name_data and name_data.IsValid(),
                'Platform has a name')
            self.assertEqual(
                name_data.GetType(), lldb.eStructuredDataTypeString,
                'Platform name is a string')
            self.assertTrue(
                desc_data and desc_data.IsValid(),
                'Platform has a description')
            self.assertEqual(
                desc_data.GetType(), lldb.eStructuredDataTypeString,
                'Platform description is a string')
