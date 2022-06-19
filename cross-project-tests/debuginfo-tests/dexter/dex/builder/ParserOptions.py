# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command line options for subtools that use the builder component."""

import os

from dex.tools import Context
from dex.utils import is_native_windows


def _find_build_scripts():
    """Finds build scripts in the 'scripts' subdirectory.

    Returns:
        { script_name (str): directory (str) }
    """
    try:
        return _find_build_scripts.cached
    except AttributeError:
        scripts_directory = os.path.join(os.path.dirname(__file__), 'scripts')
        if is_native_windows():
            scripts_directory = os.path.join(scripts_directory, 'windows')
        else:
            scripts_directory = os.path.join(scripts_directory, 'posix')
        assert os.path.isdir(scripts_directory), scripts_directory
        results = {}

        for f in os.listdir(scripts_directory):
            results[os.path.splitext(f)[0]] = os.path.abspath(
                os.path.join(scripts_directory, f))

        _find_build_scripts.cached = results
        return results


def add_builder_tool_arguments(parser):
    build_group = parser.add_mutually_exclusive_group(required=True)
    build_group.add_argument('--binary',
                             metavar="<file>",
                             help='provide binary file instead of --builder')

    build_group.add_argument(
        '--builder',
        type=str,
        choices=sorted(_find_build_scripts().keys()),
        help='test builder to use')
    build_group.add_argument('--vs-solution', metavar="<file>",
        help='provide a path to an already existing visual studio solution.')
    parser.add_argument(
        '--cflags', type=str, default='', help='compiler flags')
    parser.add_argument('--ldflags', type=str, default='', help='linker flags')


def handle_builder_tool_options(context: Context) -> str:
    return _find_build_scripts()[context.options.builder]
