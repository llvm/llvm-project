# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# To run these tests:
# python -m unittest premerge_advisor_explain_test.py

import unittest
from unittest import mock

import requests

import premerge_advisor_explain


class TestGetAdvisorExplanations(unittest.TestCase):
    @mock.patch("premerge_advisor_explain.requests.get")
    def test_success(self, get):
        explanations = [{"name": "test", "explained": True, "reason": "flaky"}]
        get.return_value.json.return_value = explanations

        self.assertEqual(
            premerge_advisor_explain.get_advisor_explanations({"failures": []}),
            (explanations, True),
        )
        get.return_value.raise_for_status.assert_called_once_with()

    @mock.patch("premerge_advisor_explain.requests.get")
    def test_timeout_fails_open(self, get):
        get.side_effect = requests.Timeout("timed out")

        with mock.patch("sys.stderr"):
            self.assertEqual(
                premerge_advisor_explain.get_advisor_explanations({"failures": []}),
                ([], False),
            )

    @mock.patch("premerge_advisor_explain.requests.get")
    def test_http_error_fails_open(self, get):
        get.return_value.raise_for_status.side_effect = requests.HTTPError(
            "service unavailable"
        )

        with mock.patch("sys.stderr"):
            self.assertEqual(
                premerge_advisor_explain.get_advisor_explanations({"failures": []}),
                ([], False),
            )


class TestMain(unittest.TestCase):
    @mock.patch("builtins.open", mock.mock_open())
    @mock.patch("premerge_advisor_explain.get_comment", return_value={"body": "report"})
    @mock.patch(
        "premerge_advisor_explain.generate_test_report_lib.generate_report",
        return_value=("report", False),
    )
    @mock.patch(
        "premerge_advisor_explain.generate_test_report_lib.get_failures",
        return_value={},
    )
    @mock.patch(
        "premerge_advisor_explain.generate_test_report_lib.load_info_from_files",
        return_value=([], []),
    )
    @mock.patch(
        "premerge_advisor_explain.get_advisor_explanations",
        return_value=([], False),
    )
    def test_advisor_failure_allows_build_to_succeed(self, *_):
        self.assertTrue(premerge_advisor_explain.main("commit", [], "token", 123, 1))


if __name__ == "__main__":
    unittest.main()
