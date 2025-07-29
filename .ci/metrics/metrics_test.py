# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for metrics.py"""

from dataclasses import dataclass
import requests
import unittest
import unittest.mock

import metrics


class TestMetrics(unittest.TestCase):
    def test_upload_gauge_metric(self):
        """Test that we can upload a gauge metric correctly.

        Also verify that we pass around parameters like API keys and user IDs
        correctly to the HTTP POST request.
        """
        test_metrics = [metrics.GaugeMetric("gauge_test", 5, 1000)]
        return_value = requests.Response()
        return_value.status_code = 204
        with unittest.mock.patch(
            "requests.post", return_value=return_value
        ) as post_mock:
            metrics.upload_metrics(test_metrics, "test_userid", "test_api_key")
            self.assertSequenceEqual(post_mock.call_args.args, [metrics.GRAFANA_URL])
            self.assertEqual(
                post_mock.call_args.kwargs["data"], "gauge_test value=5 1000"
            )
            self.assertEqual(
                post_mock.call_args.kwargs["auth"], ("test_userid", "test_api_key")
            )

    def test_upload_job_metric(self):
        """Test that we can upload a job metric correctly."""
        test_metrics = [
            metrics.JobMetrics("test_job", 5, 10, 1, 1000, 7, "test_workflow")
        ]
        return_value = requests.Response()
        return_value.status_code = 204
        with unittest.mock.patch(
            "requests.post", return_value=return_value
        ) as post_mock:
            metrics.upload_metrics(test_metrics, "test_userid", "test_aoi_key")
            self.assertEqual(
                post_mock.call_args.kwargs["data"],
                "test_job queue_time=5,run_time=10,status=1 1000",
            )

    def test_upload_unknown_metric(self):
        """Test we report an error if we encounter an unknown metric type."""

        @dataclass
        class FakeMetric:
            fake_data: str

        test_metrics = [FakeMetric("test")]

        with self.assertRaises(ValueError):
            metrics.upload_metrics(test_metrics, "test_userid", "test_api_key")

    def test_bad_response_code(self):
        """Test that we gracefully handle HTTP response errors."""
        test_metrics = [metrics.GaugeMetric("gauge_test", 5, 1000)]
        return_value = requests.Response()
        return_value.status_code = 403
        # Just assert that we continue running here and do not raise anything.
        with unittest.mock.patch("requests.post", return_value=return_value) as _:
            metrics.upload_metrics(test_metrics, "test_userid", "test_api_key")


if __name__ == "__main__":
    unittest.main()
