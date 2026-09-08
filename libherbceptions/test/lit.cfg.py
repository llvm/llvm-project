# -*- Python -*-
#===--- lit.cfg.py - libherbceptions lit test configuration ----------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#

import os
import platform
import lit.formats

# The name of this test suite.
config.name = "libherbceptions"

# Test format: use the built-in shell test format.
config.test_format = lit.formats.ShTest(True)

# Suffixes for lit test files.
config.suffixes = [".cpp"]

# Exclusions: none for now.
config.excludes = []

# Available features.
config.available_features = []
if platform.system() == "Windows":
    config.available_features.append("windows")
if platform.system() == "Linux":
    config.available_features.append("linux")
if platform.system() == "Darwin":
    config.available_features.append("darwin")

# Define compiler substitutions.
cxx = lit_config.params.get("cxx", "clang++")
config.substitutions.append(("%cxx", cxx))

# Find FileCheck.
filecheck = lit_config.params.get("filecheck", "")
if not filecheck:
    filecheck_path = lit_config.params.get("llvm_tools_dir", "")
    if filecheck_path:
        filecheck = os.path.join(filecheck_path, "FileCheck")
        if platform.system() == "Windows":
            filecheck += ".exe"
    else:
        filecheck = "FileCheck"
config.substitutions.append(("%FileCheck", filecheck))

# Herbceptions include and library paths.
herbceptions_src_root = getattr(config, "herbceptions_src_root", None)
herbceptions_obj_root = getattr(config, "herbceptions_obj_root", None)

if herbceptions_src_root:
    config.substitutions.append(("%herbceptions_include", str(herbceptions_src_root / "include")))
    config.substitutions.append(("%herbceptions_src", str(herbceptions_src_root)))
else:
    config.substitutions.append(("%herbceptions_include", ""))
    config.substitutions.append(("%herbceptions_src", ""))

if herbceptions_obj_root:
    config.substitutions.append(("%herbceptions_lib", str(herbceptions_obj_root)))
else:
    config.substitutions.append(("%herbceptions_lib", ""))

# Common flags for herbceptions tests.
herbceptions_flags = "-fherbceptions -fexceptions"
config.substitutions.append(("%herbceptions_flags", herbceptions_flags))

# Environment variables.
config.environment["PATH"] = os.pathsep.join([
    os.path.join(str(herbceptions_obj_root), "bin") if herbceptions_obj_root else "",
    os.environ.get("PATH", ""),
])
