# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from .. import config_shims
from .shim import Shim, error_no_config


def run_shim(invoke_ssaf_tools, argv, env=None):
    workflow_config, pass_config = config_shims.bootstrap_from_env(env)
    if workflow_config is None:
        return error_no_config()

    shim = Shim(workflow_config, pass_config, argv)
    cwd = os.getcwd()
    if not shim.skip_binary_tool_invocation:
        status = shim.run_bin_tool(shim.bin_tool_path(), cwd)
        if status != 0:
            return status

    return shim.protect(lambda: invoke_ssaf_tools(shim, cwd))
