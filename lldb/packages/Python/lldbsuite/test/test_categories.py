"""
Provides definitions for various lldb test categories
"""

from __future__ import absolute_import
from __future__ import print_function

# System modules
import sys

# Third-party modules

# LLDB modules
from lldbsuite.support import gmodules


debug_info_categories = [
    'dwarf', 'dwo', 'dsym', 'gmodules'
]

all_categories = {
    'swiftpr': 'Tests that may run as a part of Swift pull-request testing',
    'swift-history': 'Tests that work even when compiled older versions of swiftc',
    'dataformatters': 'Tests related to the type command and the data formatters subsystem',
    'dwarf': 'Tests that can be run with DWARF debug information',
    'dwo': 'Tests that can be run with DWO debug information',
    'dsym': 'Tests that can be run with DSYM debug information',
    'gmodules': 'Tests that can be run with -gmodules debug information',
    'expression': 'Tests related to the expression parser',
    'libc++': 'Test for libc++ data formatters',
    'libstdcxx': 'Test for libstdcxx data formatters',
    'objc': 'Tests related to the Objective-C programming language support',
    'pyapi': 'Tests related to the Python API',
    'basic_process': 'Basic process execution sniff tests.',
    'cmdline': 'Tests related to the LLDB command-line interface',
    'dyntype': 'Tests related to dynamic type support',
    'stresstest': 'Tests related to stressing lldb limits',
    'flakey': 'Flakey test cases, i.e. tests that do not reliably pass at each execution',
    'lldb-mi': 'lldb-mi tests',
    'frame-diagnose': 'Frame diagnose tests',
    'darwin-log': 'Darwin log tests',
    'watchpoint': 'Watchpoint-related tests',
}


def unique_string_match(yourentry, list):
    candidate = None
    for item in list:
        if not item.startswith(yourentry):
            continue
        if candidate:
            return None
        candidate = item
    return candidate


def is_supported_on_platform(category, platform, compiler_path):
    if category == "dwo":
        # -gsplit-dwarf is not implemented by clang on Windows.
        return platform in ["linux", "freebsd"]
    elif category == "dsym":
        return platform in ["darwin", "macosx", "ios", "watchos", "tvos", "bridgeos"]
    elif category == "gmodules":
        # First, check to see if the platform can even support gmodules.
        if platform not in ["freebsd", "darwin", "macosx", "ios", "watchos", "tvos", "bridgeos"]:
            return False
        return gmodules.is_compiler_clang_with_gmodules(compiler_path)
    return True


def validate(categories, exact_match):
    """
    For each category in categories, ensure that it's a valid category (if exact_match is false,
    unique prefixes are also accepted). If a category is invalid, print a message and quit.
       If all categories are valid, return the list of categories. Prefixes are expanded in the
       returned list.
    """
    result = []
    for category in categories:
        origCategory = category
        if category not in all_categories and not exact_match:
            category = unique_string_match(category, all_categories)
        if (category not in all_categories) or category is None:
            print(
                "fatal error: category '" +
                origCategory +
                "' is not a valid category")
            print("if you have added a new category, please edit test_categories.py, adding your new category to all_categories")
            print("else, please specify one or more of the following: " +
                  str(list(all_categories.keys())))
            sys.exit(1)
        result.append(category)
    return result
