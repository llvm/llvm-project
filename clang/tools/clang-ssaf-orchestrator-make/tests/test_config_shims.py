# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import tempfile
import unittest

from clang_ssaf_orchestrator_make import config, config_shims

from . import configs


class Serialization(unittest.TestCase):
    def test_nothing_is_read_outside_an_orchestrator_run(self):
        self.assertEqual((None, None), config_shims.bootstrap_from_env({}))


class ReadingBack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _written(self, **overrides):
        document = configs.workflow_config().to_json()
        document.update(configs.analysis_config().to_json())
        document.update(overrides)
        return document

    def test_an_analysis_document_rebuilds_an_analysis_pass(self):
        _, pass_config = config_shims.from_json(self._written())
        self.assertIsInstance(pass_config, config.SSAFAnalysisPassConfig)

    def test_a_document_with_no_pass_kind_is_refused(self):
        document = self._written()
        del document["pass_kind"]
        with self.assertRaises(KeyError):
            config_shims.from_json(document)

    def test_an_unknown_pass_kind_is_refused(self):
        with self.assertRaises(ValueError):
            config_shims.from_json(self._written(pass_kind="something-else"))

    def test_an_unexpected_key_is_refused(self):
        with self.assertRaises(TypeError):
            config_shims.from_json(self._written(unexpected="x"))

    def test_nothing_is_read_when_the_variable_is_empty(self):
        self.assertEqual(
            (None, None),
            config_shims.bootstrap_from_env({config.SSAF_CONFIG_ENV_VAR: ""}),
        )

    def test_a_variable_naming_a_file_that_is_not_there(self):
        with self.assertRaises(OSError):
            config_shims.bootstrap_from_env(
                {config.SSAF_CONFIG_ENV_VAR: os.path.join(self.tmp, "absent")}
            )


if __name__ == "__main__":
    unittest.main()
