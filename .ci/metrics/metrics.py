# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Collects Github metrics and uploads them to Grafana.

This script contains machinery that will pull metrics periodically from Github
about workflow runs. It will upload the collected metrics to the specified
Grafana instance.
"""

import collections
import datetime
import github
import logging
import os
import requests
import time

from dataclasses import dataclass
from github import Auth
from github import Github

GRAFANA_URL = (
    "https://influx-prod-13-prod-us-east-0.grafana.net/api/v1/push/influx/write"
)
SCRAPE_INTERVAL_SECONDS = 5 * 60

# Lists the Github workflows we want to track. Maps the Github job name to
# the metric name prefix in grafana.
# This metric name is also used as a key in the job->name map.
GITHUB_WORKFLOW_TO_TRACK = {
    "CI Checks": "github_llvm_premerge_checks",
    "Build and Test libc++": "github_libcxx_premerge_checks",
}

# Lists the Github jobs to track for a given workflow. The key is the stable
# name (metric name) of the workflow (see GITHUB_WORKFLOW_TO_TRACK).
# Each value is a map to link the github job name to the corresponding metric
# name.
GITHUB_JOB_TO_TRACK = {
    "github_llvm_premerge_checks": {
        "Build and Test Linux": "premerge_linux",
        "Build and Test Windows": "premerge_windows",
    },
    "github_libcxx_premerge_checks": {
        "stage1": "premerge_libcxx_stage1",
        "stage2": "premerge_libcxx_stage2",
        "stage3": "premerge_libcxx_stage3",
    },
}

# The number of workflows to pull when sampling Github workflows.
# - Github API filtering is broken: we cannot apply any filtering:
# - See https://github.com/orgs/community/discussions/86766
# - A workflow can complete before another workflow, even when starting later.
# - We don't want to sample the same workflow twice.
#
# This means we essentially have a list of workflows sorted by creation date,
# and that's all we can deduce from it. So for each iteration, we'll blindly
# process the last N workflows.
GITHUB_WORKFLOWS_MAX_PROCESS_COUNT = 2000
# Second reason for the cut: reaching a workflow older than X.
# This means we will miss long-tails (exceptional jobs running for more than
# X hours), but that's also the case with the count cutoff above.
# Only solution to avoid missing any workflow would be to process the complete
# list, which is not possible.
GITHUB_WORKFLOW_MAX_CREATED_AGE_HOURS = 8

# Grafana will fail to insert any metric older than ~2 hours (value determined
# by trial and error).
GRAFANA_METRIC_MAX_AGE_MN = 120

@dataclass
class JobMetrics:
    job_name: str
    queue_time: int
    run_time: int
    status: int
    created_at_ns: int
    started_at_ns: int
    completed_at_ns: int
    workflow_id: int
    workflow_name: str


@dataclass
class GaugeMetric:
    name: str
    value: int
    time_ns: int


@dataclass
class AggregateMetric:
    aggregate_name: str
    aggregate_queue_time: int
    aggregate_run_time: int
    aggregate_status: int
    completed_at_ns: int
    workflow_id: int


def _construct_aggregate(ag_name: str, job_list: list[JobMetrics]) -> AggregateMetric:
    """Create a libc++ AggregateMetric from a list of libc++ JobMetrics

    How aggregates are computed:
    queue time: Time from when first job in group is created until last job in
                group has started.
    run time: Time from when first job in group starts running until last job
              in group finishes running.
    status: logical 'and' of all the job statuses in the group.

    Args:
      ag_name: The name for this particular AggregateMetric
      job_list: This list of JobMetrics to be combined into the AggregateMetric.
        The input list should contain all (and only!) the libc++ JobMetrics
        for a particular stage and a particular workflow_id.

    Returns:
      Returns the AggregateMetric constructed from the inputs.
    """

    # Initialize the aggregate values
    earliest_create = job_list[0].created_at_ns
    earliest_start = job_list[0].started_at_ns
    earliest_complete = job_list[0].completed_at_ns
    latest_start = job_list[0].started_at_ns
    latest_complete = job_list[0].completed_at_ns
    ag_status = job_list[0].status
    ag_workflow_id = job_list[0].workflow_id

    # Go through rest of jobs for this workflow id, if any, updating stats
    for job in job_list[1:]:
        # Update the status
        ag_status = ag_status and job.status
        # Get the earliest & latest times
        if job.created_at_ns < earliest_create:
            earliest_create = job.created_at_ns
        if job.completed_at_ns < earliest_complete:
            earliest_complete = job.completed_at_ns
        if job.started_at_ns > latest_start:
            latest_start = job.started_at_ns
        if job.started_at_ns < earliest_start:
            earliest_start = job.started_at_ns
        if job.completed_at_ns > latest_complete:
            latest_complete = job.completed_at_ns

    # Compute aggregate run time (in seconds, not ns)
    ag_run_time = (latest_complete - earliest_start) / 1000000000
    # Compute aggregate queue time (in seconds, not ns)
    ag_queue_time = (latest_start - earliest_create) / 1000000000
    # Append the aggregate metrics to the workflow metrics list.
    return AggregateMetric(
        ag_name, ag_queue_time, ag_run_time, ag_status, latest_complete, ag_workflow_id
    )


def create_and_append_libcxx_aggregates(workflow_metrics: list[JobMetrics]):
    """Find libc++ JobMetric entries and create aggregate metrics for them.

    Sort the libc++ JobMetric entries by workflow id, and for each workflow
    id group them by stages. Call _construct_aggregate to reate an aggregate
    metric for each stage for each unique workflow id. Append each aggregate
    metric to the input workflow_metrics list.

     Args:
      workflow_metrics: A list of JobMetrics entries collected so far.
    """
    # Separate the jobs by workflow_id. Only look at JobMetrics entries.
    aggregate_data = dict()
    for job in workflow_metrics:
        # Only want to look at JobMetrics
        if not isinstance(job, JobMetrics):
            continue
        # Only want libc++ jobs.
        if job.workflow_name != "Build and Test libc++":
            continue
        if job.workflow_id not in aggregate_data.keys():
            aggregate_data[job.workflow_id] = [job]
        else:
            aggregate_data[job.workflow_id].append(job)

    # Go through each aggregate_data list (workflow id) and find all the
    # needed data
    for ag_workflow_id in aggregate_data:
        job_list = aggregate_data[ag_workflow_id]
        stage1_jobs = list()
        stage2_jobs = list()
        stage3_jobs = list()
        # sort jobs into stage1, stage2, & stage3.
        for job in job_list:
            if job.job_name.find("stage1") > 0:
                stage1_jobs.append(job)
            elif job.job_name.find("stage2") > 0:
                stage2_jobs.append(job)
            elif job.job_name.find("stage3") > 0:
                stage3_jobs.append(job)

        if len(stage1_jobs) > 0:
            aggregate = _construct_aggregate(
                "github_libcxx_premerge_checks_stage1_aggregate", stage1_jobs
            )
            workflow_metrics.append(aggregate)
        if len(stage2_jobs) > 0:
            aggregate = _construct_aggregate(
                "github_libcxx_premerge_checks_stage2_aggregate", stage2_jobs
            )
            workflow_metrics.append(aggregate)
        if len(stage3_jobs) > 0:
            aggregate = _construct_aggregate(
                "github_libcxx_premerge_checks_stage3_aggregate", stage3_jobs
            )
            workflow_metrics.append(aggregate)


def clean_up_libcxx_job_name(old_name: str) -> str:
    """Convert libcxx job names to generically legal strings.

    Args:
      old_name: A string with the full name of the libc++ test that was run.

    Returns:
      Returns the input string with characters that might not be acceptable
        in some indentifier strings replaced with safer characters.

    Take a name like 'stage1 (generic-cxx03, clang-22, clang++-22)'
    and convert it to 'stage1_generic_cxx03__clang_22__clangxx_22'.
    (Remove parentheses; replace commas, hyphens and spaces with
    underscores; replace '+' with 'x'.)
    """
    # Names should have exactly one set of parentheses, so break on that. If
    # they don't have any parentheses, then don't update them at all.
    if old_name.find("(") == -1:
        return old_name
    stage, remainder = old_name.split("(")
    stage = stage.strip()
    if remainder[-1] == ")":
        remainder = remainder[:-1]
    remainder = remainder.replace("-", "_")
    remainder = remainder.replace(",", "_")
    remainder = remainder.replace(" ", "_")
    remainder = remainder.replace("+", "x")
    new_name = stage + "_" + remainder
    return new_name

def github_get_metrics(
    github_repo: github.Repository, last_workflows_seen_as_completed: set[int]
) -> tuple[list[JobMetrics], int]:
    """Gets the metrics for specified Github workflows.

    This function takes in a list of workflows to track, and optionally the
    workflow ID of the last tracked invocation. It grabs the relevant data
    from Github, returning it to the caller.
    If the last_seen_workflow parameter is None, this returns no metrics, but
    returns the id of the most recent workflow.

    Args:
      github_repo: A github repo object to use to query the relevant information.
      last_seen_workflow: the last workflow this function processed.

    Returns:
      Returns a tuple with 2 elements:
        - a list of JobMetrics objects, one per processed job.
        - the ID of the most recent processed workflow run.
    """
    workflow_metrics = []
    queued_count = collections.Counter()
    running_count = collections.Counter()

    # Initialize all the counters to 0 so we report 0 when no job is queued
    # or running.
    for wf_name, wf_metric_name in GITHUB_WORKFLOW_TO_TRACK.items():
        for job_name, job_metric_name in GITHUB_JOB_TO_TRACK[wf_metric_name].items():
            queued_count[wf_metric_name + "_" + job_metric_name] = 0
            running_count[wf_metric_name + "_" + job_metric_name] = 0

    # The list of workflows this iteration will process.
    # MaxSize = GITHUB_WORKFLOWS_MAX_PROCESS_COUNT
    workflow_seen_as_completed = set()

    # Since we process a fixed count of workflows, we want to know when
    # the depth is too small and if we miss workflows.
    # E.g.: is there was more than N workflows int last 2 hours.
    # To monitor this, we'll log the age of the oldest workflow processed,
    # and setup alterting in Grafana to help us adjust this depth.
    oldest_seen_workflow_age_mn = None

    # Do not apply any filters to this query.
    # See https://github.com/orgs/community/discussions/86766
    # Applying filters like `status=completed` will break pagination, and
    # return a non-sorted and incomplete list of workflows.
    i = 0
    for task in iter(github_repo.get_workflow_runs()):
        # Max depth reached, stopping.
        if i >= GITHUB_WORKFLOWS_MAX_PROCESS_COUNT:
            break
        i += 1

        workflow_age_mn = (
            datetime.datetime.now(datetime.timezone.utc) - task.created_at
        ).total_seconds() / 60
        oldest_seen_workflow_age_mn = workflow_age_mn
        # If we reach a workflow older than X, stop.
        if workflow_age_mn > GITHUB_WORKFLOW_MAX_CREATED_AGE_HOURS * 60:
            break

        # This workflow is not interesting to us.
        if task.name not in GITHUB_WORKFLOW_TO_TRACK:
            continue

        libcxx_testing = False
        if task.name == "Build and Test libc++":
            libcxx_testing = True

        if task.status == "completed":
            workflow_seen_as_completed.add(task.id)

        # This workflow has already been seen completed in the previous run.
        if task.id in last_workflows_seen_as_completed:
            continue

        name_prefix = GITHUB_WORKFLOW_TO_TRACK[task.name]
        for job in task.jobs():
            if libcxx_testing:
                # We're not running macos or windows libc++ tests on our
                # infrastructure.
                if job.name.find("macos") != -1 or job.name.find("windows") != -1:
                    continue
            # This job is not interesting to us.
            elif job.name not in GITHUB_JOB_TO_TRACK[name_prefix]:
                continue

            if libcxx_testing:
                name_suffix = clean_up_libcxx_job_name(job.name)
            else:
                name_suffix = GITHUB_JOB_TO_TRACK[name_prefix][job.name]
            metric_name = name_prefix + "_" + name_suffix

            if task.status != "completed":
                if job.status == "queued":
                    queued_count[metric_name] += 1
                elif job.status == "in_progress":
                    running_count[metric_name] += 1
                continue

            job_result = int(job.conclusion == "success" or job.conclusion == "skipped")

            created_at = job.created_at
            started_at = job.started_at
            completed_at = job.completed_at

            # GitHub API can return results where the started_at is slightly
            # later then the created_at (or completed earlier than started).
            # This would cause a -23h59mn delta, which will show up as +24h
            # queue/run time on grafana.
            if started_at < created_at:
                logging.info(
                    "Workflow {} started before being created.".format(task.id)
                )
                queue_time = datetime.timedelta(seconds=0)
            else:
                queue_time = started_at - created_at
            if completed_at < started_at:
                logging.info("Workflow {} finished before starting.".format(task.id))
                run_time = datetime.timedelta(seconds=0)
            else:
                run_time = completed_at - started_at

            if run_time.seconds == 0:
                continue

            # Grafana will refuse to ingest metrics older than ~2 hours, so we
            # should avoid sending historical data.
            metric_age_mn = (
                datetime.datetime.now(datetime.timezone.utc) - completed_at
            ).total_seconds() / 60
            if metric_age_mn > GRAFANA_METRIC_MAX_AGE_MN:
                logging.warning(
                    f"Job {job.id} from workflow {task.id} dropped due"
                    + f" to staleness: {metric_age_mn}mn old."
                )
                continue

            logging.info(f"Adding a job metric for job {job.id} in workflow {task.id}")
            # The completed_at_ns timestamp associated with the event is
            # expected by Grafana to be in nanoseconds. Because we do math using
            # all three times (when creating libc++ aggregates), we need them
            # all to be in nanoseconds, even though created_at and started_at
            # are not returned to Grafana.
            created_at_ns = int(created_at.timestamp()) * 10**9
            started_at_ns = int(started_at.timestamp()) * 10**9
            completed_at_ns = int(completed_at.timestamp()) * 10**9
            workflow_metrics.append(
                JobMetrics(
                    metric_name,
                    queue_time.seconds,
                    run_time.seconds,
                    job_result,
                    created_at_ns,
                    started_at_ns,
                    completed_at_ns,
                    task.id,
                    task.name,
                )
            )

    # Finished collecting the JobMetrics for all jobs; now create the
    # aggregates for any libc++ jobs.
    create_and_append_libcxx_aggregates(workflow_metrics)

    for name, value in queued_count.items():
        workflow_metrics.append(
            GaugeMetric(f"workflow_queue_size_{name}", value, time.time_ns())
        )
    for name, value in running_count.items():
        workflow_metrics.append(
            GaugeMetric(f"running_workflow_count_{name}", value, time.time_ns())
        )

    # Always send a hearbeat metric so we can monitor is this container is still able to log to Grafana.
    workflow_metrics.append(
        GaugeMetric("metrics_container_heartbeat", 1, time.time_ns())
    )

    # Log the oldest workflow we saw, allowing us to monitor if the processing
    # depth is correctly set-up.
    if oldest_seen_workflow_age_mn is not None:
        workflow_metrics.append(
            GaugeMetric(
                "github_oldest_processed_workflow_mn",
                oldest_seen_workflow_age_mn,
                time.time_ns(),
            )
        )
    return workflow_metrics, workflow_seen_as_completed


def upload_metrics(workflow_metrics, metrics_userid, api_key):
    """Upload metrics to Grafana.

    Takes in a list of workflow metrics and then uploads them to Grafana
    through a REST request.

    Args:
      workflow_metrics: A list of metrics to upload to Grafana.
      metrics_userid: The userid to use for the upload.
      api_key: The API key to use for the upload.
    """

    if len(workflow_metrics) == 0:
        logging.info("No metrics found to upload.")
        return

    metrics_batch = []
    for workflow_metric in workflow_metrics:
        if isinstance(workflow_metric, GaugeMetric):
            name = workflow_metric.name.lower().replace(" ", "_")
            metrics_batch.append(
                f"{name} value={workflow_metric.value} {workflow_metric.time_ns}"
            )
        elif isinstance(workflow_metric, JobMetrics):
            name = workflow_metric.job_name.lower().replace(" ", "_")
            metrics_batch.append(
                f"{name} queue_time={workflow_metric.queue_time},run_time={workflow_metric.run_time},status={workflow_metric.status} {workflow_metric.completed_at_ns}"
            )
        elif isinstance(workflow_metric, AggregateMetric):
            name = workflow_metric.aggregate_name.lower().replace(" ", "_")
            metrics_batch.append(
                f"{name} queue_time={workflow_metric.aggregate_queue_time},run_time={workflow_metric.aggregate_run_time},status={workflow_metric.aggregate_status} {workflow_metric.completed_at_ns}"
            )
        else:
            raise ValueError(
                f"Unsupported object type {type(workflow_metric)}: {str(workflow_metric)}"
            )

    request_data = "\n".join(metrics_batch)
    response = requests.post(
        GRAFANA_URL,
        headers={"Content-Type": "text/plain"},
        data=request_data,
        auth=(metrics_userid, api_key),
    )

    if response.status_code < 200 or response.status_code >= 300:
        logging.info(f"Failed to submit data to Grafana: {response.status_code}")


def main():
    # Authenticate with Github
    github_auth = Auth.Token(os.environ["GITHUB_TOKEN"])
    grafana_api_key = os.environ["GRAFANA_API_KEY"]
    grafana_metrics_userid = os.environ["GRAFANA_METRICS_USERID"]

    # The last workflow this script processed.
    # Because the Github queries are broken, we'll simply log a 'processed'
    # bit for the last COUNT_TO_PROCESS workflows.
    gh_last_workflows_seen_as_completed = set()

    # Enter the main loop. Every five minutes we wake up and dump metrics for
    # the relevant jobs.
    while True:
        github_object = Github(auth=github_auth)
        github_repo = github_object.get_repo("llvm/llvm-project")

        gh_metrics, gh_last_workflows_seen_as_completed = github_get_metrics(
            github_repo, gh_last_workflows_seen_as_completed
        )

        upload_metrics(gh_metrics, grafana_metrics_userid, grafana_api_key)
        logging.info(f"Uploaded {len(gh_metrics)} metrics")

        time.sleep(SCRAPE_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
