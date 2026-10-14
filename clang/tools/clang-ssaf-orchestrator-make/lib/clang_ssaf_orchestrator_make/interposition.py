# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from . import config

SHIM_PROGRAM = "shim"


def create_shim_links(shim_link_dir_path, installed_shim_dir_path):
    os.makedirs(shim_link_dir_path)
    program = os.path.join(installed_shim_dir_path, SHIM_PROGRAM)
    if not os.access(program, os.X_OK):
        raise OSError("shim is not executable: %s" % program)
    for name in config.SHIM_FOR_LINK_NAME:
        os.symlink(program, os.path.join(shim_link_dir_path, name))
    return shim_link_dir_path


def make_environment(serialized_config_path, shim_link_dir_path, env):
    result = dict(env)
    result[config.SSAF_CONFIG_ENV_VAR] = serialized_config_path
    build_path = result.get("PATH", "")
    result["PATH"] = (
        os.pathsep.join([shim_link_dir_path, build_path])
        if build_path
        else shim_link_dir_path
    )
    return result
