# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# To run these tests:
# python -m unittest premerge_advisor_upload_test.py

import unittest
from unittest import mock

import requests

import premerge_advisor_upload


class TestUploadFailureInfo(unittest.TestCase):
    @mock.patch("premerge_advisor_upload.requests.post")
    def test_success(self, post):
        premerge_advisor_upload.upload_failure_info({"failures": []})

        self.assertEqual(
            post.call_count, len(premerge_advisor_upload.PREMERGE_ADVISOR_URLS)
        )
        self.assertEqual(
            post.return_value.raise_for_status.call_count,
            len(premerge_advisor_upload.PREMERGE_ADVISOR_URLS),
        )

    @mock.patch("premerge_advisor_upload.requests.post")
    def test_request_failures_are_ignored(self, post):
        post.side_effect = requests.Timeout("timed out")

        with mock.patch("sys.stderr"):
            premerge_advisor_upload.upload_failure_info({"failures": []})

        self.assertEqual(
            post.call_count, len(premerge_advisor_upload.PREMERGE_ADVISOR_URLS)
        )


if __name__ == "__main__":
    unittest.main()
