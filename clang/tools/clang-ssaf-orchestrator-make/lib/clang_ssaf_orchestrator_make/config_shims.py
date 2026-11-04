# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os

from . import config


def from_json(written):
    shared = config.SSAFWorkflowConfig.written_field_names()
    workflow_config = config.SSAFWorkflowConfig(
        **{key: value for key, value in written.items() if key in shared}
    )
    own = {
        key: value
        for key, value in written.items()
        if key not in shared and key != "pass_kind"
    }
    config.SSAFPassKind(written["pass_kind"])
    return workflow_config, config.SSAFAnalysisPassConfig(**own)


def bootstrap_from_env(env=None):
    env = os.environ if env is None else env
    path = env.get(config.SSAF_CONFIG_ENV_VAR)
    if not path:
        return None, None
    with open(path) as handle:
        return from_json(json.load(handle))
