# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import stat
import tempfile
import unittest

from clang_ssaf_orchestrator_make import config, interposition


class ShimLinks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.installed_shim_dir_path = os.path.join(self.tmp, "libexec")
        os.makedirs(self.installed_shim_dir_path)
        self.program = os.path.join(
            self.installed_shim_dir_path, interposition.SHIM_PROGRAM
        )
        with open(self.program, "w") as handle:
            handle.write("#!/bin/sh\n")
        os.chmod(self.program, os.stat(self.program).st_mode | stat.S_IXUSR)
        self.shim_link_dir_path = os.path.join(self.tmp, "shims")

    def test_every_tool_name_is_created_every_run(self):
        interposition.create_shim_links(
            self.shim_link_dir_path, self.installed_shim_dir_path
        )
        self.assertEqual(
            sorted(config.SHIM_FOR_LINK_NAME),
            sorted(os.listdir(self.shim_link_dir_path)),
        )

    def test_every_name_is_a_link_to_the_one_program(self):
        interposition.create_shim_links(
            self.shim_link_dir_path, self.installed_shim_dir_path
        )
        for name in config.SHIM_FOR_LINK_NAME:
            link = os.path.join(self.shim_link_dir_path, name)
            self.assertTrue(os.path.islink(link), link)
            self.assertEqual(self.program, os.readlink(link))

    def test_a_program_that_cannot_be_run_is_refused(self):
        os.chmod(self.program, 0o644)
        with self.assertRaises(OSError):
            interposition.create_shim_links(
                self.shim_link_dir_path, self.installed_shim_dir_path
            )


class BuildEnvironment(unittest.TestCase):
    def test_a_build_with_no_path_sees_only_the_shim_link_directory(self):
        for env in ({}, {"PATH": ""}):
            result = interposition.make_environment("/out/c.json", "/out/shims", env)
            self.assertEqual("/out/shims", result["PATH"], env)

    def test_the_shim_directory_goes_in_front_of_the_builds_own_path(self):
        env = interposition.make_environment(
            "/out/config-analysis.json", "/out/shims", {"PATH": "/usr/bin"}
        )
        self.assertEqual(os.pathsep.join(["/out/shims", "/usr/bin"]), env["PATH"])
        self.assertEqual("/out/config-analysis.json", env[config.SSAF_CONFIG_ENV_VAR])


if __name__ == "__main__":
    unittest.main()
