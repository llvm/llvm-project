# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Does some stuff."""

import unittest

import compute_projects


class TestComputeProjects(unittest.TestCase):
    def test_llvm(self):
        env_variables = compute_projects.get_env_variables(
            ["llvm/CMakeLists.txt"], "Linux"
        )
        self.assertEqual(
            env_variables["projects_to_build"],
            "bolt;clang;clang-tools-extra;flang;lld;lldb;llvm;mlir;polly",
        )
        self.assertEqual(
            env_variables["project_check_targets"],
            "check-bolt check-clang check-clang-tools check-flang check-lld check-lldb check-llvm check-mlir check-polly",
        )
        self.assertEqual(
            env_variables["runtimes_to_build"], "libcxx;libcxxabi;libunwind"
        )
        self.assertEqual(
            env_variables["runtimes_check_targets"],
            "check-cxx check-cxxabi check-unwind",
        )

    def test_llvm_windows(self):
        env_variables = compute_projects.get_env_variables(
            ["llvm/CMakeLists.txt"], "Windows"
        )
        self.assertEqual(
            env_variables["projects_to_build"],
            "clang;clang-tools-extra;lld;llvm;mlir;polly",
        )
        self.assertEqual(
            env_variables["project_check_targets"],
            "check-clang check-clang-tools check-lld check-llvm check-mlir check-polly",
        )
        self.assertEqual(
            env_variables["runtimes_to_build"], "libcxx;libcxxabi;libunwind"
        )
        self.assertEqual(
            env_variables["runtimes_check_targets"],
            "check-cxx check-cxxabi check-unwind",
        )

    def test_llvm_mac(self):
        env_variables = compute_projects.get_env_variables(
            ["llvm/CMakeLists.txt"], "Darwin"
        )
        self.assertEqual(
            env_variables["projects_to_build"],
            "clang;clang-tools-extra;lld;llvm;mlir",
        )
        self.assertEqual(
            env_variables["project_check_targets"],
            "check-clang check-clang-tools check-lld check-llvm check-mlir",
        )
        self.assertEqual(
            env_variables["runtimes_to_build"], "libcxx;libcxxabi;libunwind"
        )
        self.assertEqual(
            env_variables["runtimes_check_targets"],
            "check-cxx check-cxxabi check-unwind",
        )

    def test_clang(self):
        env_variables = compute_projects.get_env_variables(
            ["clang/CMakeLists.txt"], "Linux"
        )
        self.assertEqual(
            env_variables["projects_to_build"],
            "clang;clang-tools-extra;compiler-rt;lld;llvm",
        )
        self.assertEqual(
            env_variables["project_check_targets"],
            "check-clang check-clang-tools check-compiler-rt",
        )
        self.assertEqual(
            env_variables["runtimes_to_build"], "libcxx;libcxxabi;libunwind"
        )
        self.assertEqual(
            env_variables["runtimes_check_targets"],
            "check-cxx check-cxxabi check-unwind",
        )

    def test_clang_windows(self):
        env_variables = compute_projects.get_env_variables(
            ["clang/CMakeLists.txt"], "Windows"
        )
        self.assertEqual(
            env_variables["projects_to_build"], "clang;clang-tools-extra;llvm"
        )
        self.assertEqual(
            env_variables["project_check_targets"], "check-clang check-clang-tools"
        )
        self.assertEqual(
            env_variables["runtimes_to_build"], "libcxx;libcxxabi;libunwind"
        )
        self.assertEqual(
            env_variables["runtimes_check_targets"],
            "check-cxx check-cxxabi check-unwind",
        )

    def test_bolt(self):
        env_variables = compute_projects.get_env_variables(
            ["bolt/CMakeLists.txt"], "Linux"
        )
        self.assertEqual(env_variables["projects_to_build"], "bolt;clang;lld;llvm")
        self.assertEqual(env_variables["project_check_targets"], "check-bolt")
        self.assertEqual(env_variables["runtimes_to_build"], "")
        self.assertEqual(env_variables["runtimes_check_targets"], "")

    def test_lldb(self):
        env_variables = compute_projects.get_env_variables(
            ["lldb/CMakeLists.txt"], "Linux"
        )
        self.assertEqual(env_variables["projects_to_build"], "clang;lldb;llvm")
        self.assertEqual(env_variables["project_check_targets"], "check-lldb")
        self.assertEqual(env_variables["runtimes_to_build"], "")
        self.assertEqual(env_variables["runtimes_check_targets"], "")

    def test_mlir(self):
        env_variables = compute_projects.get_env_variables(
            ["mlir/CMakeLists.txt"], "Linux"
        )
        self.assertEqual(env_variables["projects_to_build"], "clang;flang;llvm;mlir")
        self.assertEqual(
            env_variables["project_check_targets"], "check-flang check-mlir"
        )
        self.assertEqual(env_variables["runtimes_to_build"], "")
        self.assertEqual(env_variables["runtimes_check_targets"], "")

    def test_flang(self):
        env_variables = compute_projects.get_env_variables(
            ["flang/CMakeLists.txt"], "Linux"
        )
        self.assertEqual(env_variables["projects_to_build"], "clang;flang;llvm")
        self.assertEqual(env_variables["project_check_targets"], "check-flang")
        self.assertEqual(env_variables["runtimes_to_build"], "")
        self.assertEqual(env_variables["runtimes_check_targets"], "")

    def test_invalid_subproject(self):
        env_variables = compute_projects.get_env_variables(
            ["third-party/benchmark/CMakeLists.txt"], "Linux"
        )
        self.assertEqual(env_variables["projects_to_build"], "")
        self.assertEqual(env_variables["project_check_targets"], "")
        self.assertEqual(env_variables["runtimes_to_build"], "")
        self.assertEqual(env_variables["runtimes_check_targets"], "")

    def test_top_level_file(self):
        env_variables = compute_projects.get_env_variables(["README.md"], "Linux")
        self.assertEqual(env_variables["projects_to_build"], "")
        self.assertEqual(env_variables["project_check_targets"], "")
        self.assertEqual(env_variables["runtimes_to_build"], "")
        self.assertEqual(env_variables["runtimes_check_targets"], "")

    def test_exclude_runtiems_in_projects(self):
        env_variables = compute_projects.get_env_variables(
            ["libcxx/CMakeLists.txt"], "Linux"
        )
        self.assertEqual(env_variables["projects_to_build"], "")
        self.assertEqual(env_variables["project_check_targets"], "")
        self.assertEqual(env_variables["runtimes_to_build"], "")
        self.assertEqual(env_variables["runtimes_check_targets"], "")

    def test_exclude_docs(self):
        env_variables = compute_projects.get_env_variables(
            ["llvm/docs/CIBestPractices.rst"], "Linux"
        )
        self.assertEqual(env_variables["projects_to_build"], "")
        self.assertEqual(env_variables["project_check_targets"], "")
        self.assertEqual(env_variables["runtimes_to_build"], "")
        self.assertEqual(env_variables["runtimes_check_targets"], "")

    def test_exclude_gn(self):
        env_variables = compute_projects.get_env_variables(
            ["llvm/utils/gn/build/BUILD.gn"], "Linux"
        )
        self.assertEqual(env_variables["projects_to_build"], "")
        self.assertEqual(env_variables["project_check_targets"], "")
        self.assertEqual(env_variables["runtimes_to_build"], "")
        self.assertEqual(env_variables["runtimes_check_targets"], "")

    def test_ci(self):
        env_variables = compute_projects.get_env_variables(
            [".ci/compute_projects.py"], "Linux"
        )
        self.assertEqual(env_variables["projects_to_build"], "clang;lld;lldb;llvm")
        self.assertEqual(
            env_variables["project_check_targets"],
            "check-clang check-lld check-lldb check-llvm",
        )
        self.assertEqual(
            env_variables["runtimes_to_build"], "libcxx;libcxxabi;libunwind"
        )
        self.assertEqual(
            env_variables["runtimes_check_targets"],
            "check-cxx check-cxxabi check-unwind",
        )


if __name__ == "__main__":
    unittest.main()
