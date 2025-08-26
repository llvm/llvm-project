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
            metrics.JobMetrics(
                "test_job", 5, 10, 1, 1000, 1000, 1000, 7, "test_workflow"
            )
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

    def test_upload_aggregate_metric(self):
        """Test that we can upload an aggregate metric correctly."""
        test_metrics = [
            metrics.AggregateMetric("stage1_aggregate", 211, 1124, 1, 1200, 9)
        ]
        return_value = requests.Response()
        return_value.status_code = 204
        with unittest.mock.patch(
            "requests.post", return_value=return_value
        ) as post_mock:
            metrics.upload_metrics(test_metrics, "test_userid", "test_aoi_key")
            self.assertEqual(
                post_mock.call_args.kwargs["data"],
                "stage1_aggregate queue_time=211,run_time=1124,status=1 1200",
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

    def test_create_and_append_aggregate_metric_1_stage(self):
        """Test the creation of a single AggregateMetric"""
        test_metrics = [
            metrics.JobMetrics(
                "libcxx_stage1_test1",
                8,
                388,
                1,
                created_at_ns=1755697953000000000,
                started_at_ns=1755697961000000000,
                completed_at_ns=1755698349000000000,
                workflow_id=3,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage1_test2",
                107,
                357,
                1,
                created_at_ns=1755697953000000000,
                started_at_ns=1755698060000000000,
                completed_at_ns=1755698417000000000,
                workflow_id=3,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage1_test3",
                8,
                824,
                1,
                created_at_ns=1755697953000000000,
                started_at_ns=1755697961000000000,
                completed_at_ns=1755698785000000000,
                workflow_id=3,
                workflow_name="Build and Test libc++",
            ),
        ]
        metrics.create_and_append_libcxx_aggregates(test_metrics)
        self.assertEqual(len(test_metrics), 4)
        self.assertTrue(isinstance(test_metrics[-1], metrics.AggregateMetric))
        aggregate = test_metrics[-1]
        self.assertEqual(
            aggregate.aggregate_name, "github_libcxx_premerge_checks_stage1_aggregate"
        )
        self.assertEqual(aggregate.aggregate_queue_time, 107)
        self.assertEqual(aggregate.aggregate_run_time, 824)
        self.assertEqual(aggregate.aggregate_status, 1)
        self.assertEqual(aggregate.completed_at_ns, 1755698785000000000)
        self.assertEqual(aggregate.workflow_id, 3)

    def test_create_and_append_aggregate_metric_multiple_workflow_ids(self):
        """Test creation of AggregateMetric for same stage with diff workflow ids."""
        test_metrics = [
            metrics.JobMetrics(
                "libcxx_stage1_test1",
                8,
                388,
                1,
                created_at_ns=1755697953000000000,
                started_at_ns=1755697961000000000,
                completed_at_ns=1755698349000000000,
                workflow_id=3,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage1_test2",
                107,
                357,
                0,
                created_at_ns=1755697953000000000,
                started_at_ns=1755698060000000000,
                completed_at_ns=1755698417000000000,
                workflow_id=3,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage1_test3",
                8,
                824,
                1,
                created_at_ns=1755697953000000000,
                started_at_ns=1755697961000000000,
                completed_at_ns=1755698785000000000,
                workflow_id=25,
                workflow_name="Build and Test libc++",
            ),
        ]
        metrics.create_and_append_libcxx_aggregates(test_metrics)
        self.assertEqual(len(test_metrics), 5)
        self.assertTrue(isinstance(test_metrics[3], metrics.AggregateMetric))
        self.assertTrue(isinstance(test_metrics[4], metrics.AggregateMetric))
        aggregate = test_metrics[3]
        self.assertEqual(
            aggregate.aggregate_name, "github_libcxx_premerge_checks_stage1_aggregate"
        )
        self.assertEqual(aggregate.aggregate_queue_time, 107)
        self.assertEqual(aggregate.aggregate_run_time, 456)
        self.assertEqual(aggregate.aggregate_status, 0)
        self.assertEqual(aggregate.completed_at_ns, 1755698417000000000)
        self.assertEqual(aggregate.workflow_id, 3)

        aggregate = test_metrics[4]
        self.assertEqual(
            aggregate.aggregate_name, "github_libcxx_premerge_checks_stage1_aggregate"
        )
        self.assertEqual(aggregate.aggregate_queue_time, 8)
        self.assertEqual(aggregate.aggregate_run_time, 824)
        self.assertEqual(aggregate.aggregate_status, 1)
        self.assertEqual(aggregate.completed_at_ns, 1755698785000000000)
        self.assertEqual(aggregate.workflow_id, 25)

    def test_create_and_append_aggregate_metric_3_stages(self):
        """Test the creation of AggregateMetric for each of 3 stages."""
        test_metrics = [
            metrics.JobMetrics(
                "libcxx_stage1_test1",
                124,
                1454,
                1,
                created_at_ns=1755696929000000000,
                started_at_ns=1755697053000000000,
                completed_at_ns=1755698507000000000,
                workflow_id=17,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage1_test2",
                129,
                827,
                1,
                created_at_ns=1755696929000000000,
                started_at_ns=1755697058000000000,
                completed_at_ns=1755697885000000000,
                workflow_id=17,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage2_test1",
                6,
                580,
                1,
                created_at_ns=1755698507000000000,
                started_at_ns=1755698513000000000,
                completed_at_ns=1755699093000000000,
                workflow_id=17,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage2_test2",
                7,
                473,
                1,
                created_at_ns=1755698507000000000,
                started_at_ns=1755698514000000000,
                completed_at_ns=1755698987000000000,
                workflow_id=17,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage2_test3",
                7,
                820,
                1,
                created_at_ns=1755698507000000000,
                started_at_ns=1755698514000000000,
                completed_at_ns=1755699334000000000,
                workflow_id=17,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage3_test1",
                7,
                919,
                1,
                created_at_ns=1755709005000000000,
                started_at_ns=1755709012000000000,
                completed_at_ns=1755709931000000000,
                workflow_id=17,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage3_test2",
                141,
                834,
                1,
                created_at_ns=1755709005000000000,
                started_at_ns=1755709146000000000,
                completed_at_ns=1755709980000000000,
                workflow_id=17,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "libcxx_stage3_test3",
                131,
                370,
                1,
                created_at_ns=1755709005000000000,
                started_at_ns=1755709136000000000,
                completed_at_ns=1755709514000000000,
                workflow_id=17,
                workflow_name="Build and Test libc++",
            ),
        ]
        metrics.create_and_append_libcxx_aggregates(test_metrics)
        self.assertEqual(len(test_metrics), 11)
        self.assertTrue(isinstance(test_metrics[8], metrics.AggregateMetric))
        self.assertTrue(isinstance(test_metrics[9], metrics.AggregateMetric))
        self.assertTrue(isinstance(test_metrics[10], metrics.AggregateMetric))
        aggregate = test_metrics[8]
        self.assertEqual(
            aggregate.aggregate_name, "github_libcxx_premerge_checks_stage1_aggregate"
        )
        self.assertEqual(aggregate.aggregate_queue_time, 129)
        self.assertEqual(aggregate.aggregate_run_time, 1454)
        self.assertEqual(aggregate.aggregate_status, 1)
        self.assertEqual(aggregate.completed_at_ns, 1755698507000000000)
        self.assertEqual(aggregate.workflow_id, 17)

        aggregate = test_metrics[9]
        self.assertEqual(
            aggregate.aggregate_name, "github_libcxx_premerge_checks_stage2_aggregate"
        )
        self.assertEqual(aggregate.aggregate_queue_time, 7)
        self.assertEqual(aggregate.aggregate_run_time, 821)
        self.assertEqual(aggregate.aggregate_status, 1)
        self.assertEqual(aggregate.completed_at_ns, 1755699334000000000)
        self.assertEqual(aggregate.workflow_id, 17)

        aggregate = test_metrics[10]
        self.assertEqual(
            aggregate.aggregate_name, "github_libcxx_premerge_checks_stage3_aggregate"
        )
        self.assertEqual(aggregate.aggregate_queue_time, 141)
        self.assertEqual(aggregate.aggregate_run_time, 968)
        self.assertEqual(aggregate.aggregate_status, 1)
        self.assertEqual(aggregate.completed_at_ns, 1755709980000000000)
        self.assertEqual(aggregate.workflow_id, 17)

    def test_create_and_append_aggregate_metric_mixed_job_types(self):
        """Test the creation of AggregateMetric with non-lib++ jobs thrown in."""
        test_metrics = [
            metrics.JobMetrics(
                "ci_test1", 5, 10, 1, 1000, 1200, 1400, 5, "premerge_test"
            ),
            metrics.JobMetrics(
                "libcxx_stage1_test1",
                8,
                388,
                1,
                created_at_ns=1755697953000000000,
                started_at_ns=1755697961000000000,
                completed_at_ns=1755698349000000000,
                workflow_id=3,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "ci_test2", 3, 20, 1, 2000, 2200, 2400, 37, "premerge_test"
            ),
            metrics.JobMetrics(
                "libcxx_stage1_test2",
                107,
                357,
                0,
                created_at_ns=1755697953000000000,
                started_at_ns=1755698060000000000,
                completed_at_ns=1755698417000000000,
                workflow_id=3,
                workflow_name="Build and Test libc++",
            ),
            metrics.JobMetrics(
                "ci_test3", 7, 35, 1, 3000, 3200, 3400, 85, "premerge_test"
            ),
        ]
        metrics.create_and_append_libcxx_aggregates(test_metrics)
        self.assertEqual(len(test_metrics), 6)
        self.assertTrue(isinstance(test_metrics[5], metrics.AggregateMetric))
        aggregate = test_metrics[5]
        self.assertEqual(
            aggregate.aggregate_name, "github_libcxx_premerge_checks_stage1_aggregate"
        )
        self.assertEqual(aggregate.aggregate_queue_time, 107)
        self.assertEqual(aggregate.aggregate_run_time, 456)
        self.assertEqual(aggregate.aggregate_status, 0)
        self.assertEqual(aggregate.completed_at_ns, 1755698417000000000)
        self.assertEqual(aggregate.workflow_id, 3)

    def test_create_and_append_aggregate_metric_no_libcxx_jobs(self):
        """Test the creation of AggregateMetric with no libc++ jobs.

        In this case, no AggregateMetric should be created, but no
        errors or complaints should be raised.
        """
        test_metrics = [
            metrics.JobMetrics(
                "ci_test1", 5, 10, 1, 1000, 1200, 1400, 5, "premerge_test"
            ),
            metrics.JobMetrics(
                "ci_test2", 3, 20, 1, 2000, 2200, 2400, 37, "premerge_test"
            ),
            metrics.JobMetrics(
                "ci_test3", 7, 35, 1, 3000, 3200, 3400, 85, "premerge_test"
            ),
        ]
        metrics.create_and_append_libcxx_aggregates(test_metrics)
        self.assertEqual(len(test_metrics), 3)

    def test_clean_up_libcxx_job_name(self):
        """Test that we correctly update (or not) libcxx job names."""
        stage1_name = "stage1 (test1, C++-test2, my-c++-test-25)"
        stage2_name = "stage2 (generic-cxx26, clang-21, clang++21)"
        stage3_name = "stage3 (generic-cxx26, libcxx-next-runners, junk)"
        bad_name = "this is a bad name"
        out_name1 = metrics.clean_up_libcxx_job_name(stage1_name)
        self.assertEqual(out_name1, "stage1_test1__Cxx_test2__my_cxx_test_25")
        out_name2 = metrics.clean_up_libcxx_job_name(stage2_name)
        self.assertEqual(out_name2, "stage2_generic_cxx26__clang_21__clangxx21")
        out_name3 = metrics.clean_up_libcxx_job_name(stage3_name)
        self.assertEqual(out_name3, "stage3_generic_cxx26__libcxx_next_runners__junk")
        out_name4 = metrics.clean_up_libcxx_job_name(bad_name)
        self.assertEqual(out_name4, bad_name)

if __name__ == "__main__":
    unittest.main()
