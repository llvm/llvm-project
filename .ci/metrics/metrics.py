import requests
import collections
import time
import os
from dataclasses import dataclass
import sys
import logging

import github
from github import Github
from github import Auth

GRAFANA_URL = (
    "https://influx-prod-13-prod-us-east-0.grafana.net/api/v1/push/influx/write"
)
SCRAPE_INTERVAL_SECONDS = 60

# Lists the Github workflows we want to track. Maps the Github job name to
# the metric name prefix in grafana.
# This metric name is also used as a key in the job->name map.
GITHUB_WORKFLOW_TO_TRACK = {"LLVM Premerge Checks": "github_llvm_premerge_checks"}

# Lists the Github jobs to track for a given workflow. The key is the stable
# name (metric name) of the workflow (see GITHUB_WORKFLOW_TO_TRACK).
# Each value is a map to link the github job name to the corresponding metric
# name.
GITHUB_JOB_TO_TRACK = {
    "github_llvm_premerge_checks": {
        "Linux Premerge Checks (Test Only - Please Ignore Results)": "premerge_linux",
        "Windows Premerge Checks (Test Only - Please Ignore Results)": "premerge_windows",
    }
}

# The number of workflows to pull when sampling queue size & running count.
# Filtering at the query level doesn't work, and thus sampling workflow counts
# cannot be done in a clean way.
# If we miss running/queued workflows, we might want to bump this value.
GITHUB_WORKFLOWS_COUNT_FOR_SAMPLING = 200


@dataclass
class JobMetrics:
    job_name: str
    queue_time: int
    run_time: int
    status: int
    created_at_ns: int
    workflow_id: int
    workflow_name: str


@dataclass
class GaugeMetric:
    name: str
    value: int
    time_ns: int

def get_sampled_workflow_metrics(github_repo: github.Repository):
    """Gets global statistics about the Github workflow queue

    Args:
      github_repo: A github repo object to use to query the relevant information.

    Returns:
      Returns a list of GaugeMetric objects, containing the relevant metrics about
      the workflow
    """
    queued_count = collections.Counter()
    running_count = collections.Counter()

    # Do not apply any filters to this query.
    # See https://github.com/orgs/community/discussions/86766
    # Applying filters like `status=completed` will break pagination, and
    # return a non-sorted and incomplete list of workflows.
    i = 0
    for task in iter(github_repo.get_workflow_runs()):
        if i > GITHUB_WORKFLOWS_COUNT_FOR_SAMPLING:
            break
        i += 1

        if task.name not in GITHUB_WORKFLOW_TO_TRACK:
            continue

        prefix_name = GITHUB_WORKFLOW_TO_TRACK[task.name]
        for job in task.jobs():
            if job.name not in GITHUB_JOB_TO_TRACK[prefix_name]:
                continue
            suffix_name = GITHUB_JOB_TO_TRACK[prefix_name][job.name]
            metric_name = f"{prefix_name}_{suffix_name}"

            # Other states are available (pending, waiting, etc), but the meaning
            # is not documented (See #70540).
            # "queued" seems to be the info we want.
            if job.status == "queued":
                queued_count[metric_name] += 1
            elif job.status == "in_progress":
                running_count[metric_name] += 1

    workflow_metrics = []
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
    return workflow_metrics


def get_per_workflow_metrics(github_repo: github.Repository, last_seen_workflow: str):
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
    most_recent_workflow_processed = None

    # Do not apply any filters to this query.
    # See https://github.com/orgs/community/discussions/86766
    # Applying filters like `status=completed` will break pagination, and
    # return a non-sorted and incomplete list of workflows.
    for task in iter(github_repo.get_workflow_runs()):
        # Ignoring non-completed workflows.
        if task.status != "completed":
            continue

        # Record the most recent workflow we processed so this script
        # only processes it once.
        if most_recent_workflow_processed is None:
            most_recent_workflow_processed = task.id

        # This condition only happens when this script starts:
        # this is used to determine a start point. Don't return any
        # metrics, just the most recent workflow ID.
        if last_seen_workflow is None:
            break

        # This workflow has already been processed. We can stop now.
        if last_seen_workflow == task.id:
            break

        # This workflow is not interesting to us.
        if task.name not in GITHUB_WORKFLOW_TO_TRACK:
            continue

        name_prefix = GITHUB_WORKFLOW_TO_TRACK[task.name]

        for job in task.jobs():
            # This job is not interesting to us.
            if job.name not in GITHUB_JOB_TO_TRACK[name_prefix]:
                continue

            name_suffix = GITHUB_JOB_TO_TRACK[name_prefix][job.name]
            created_at = job.created_at
            started_at = job.started_at
            completed_at = job.completed_at

            job_result = int(job.conclusion == "success")
            if job_result:
                # We still might want to mark the job as a failure if one of the steps
                # failed. This is required due to use setting continue-on-error in
                # the premerge pipeline to prevent sending emails while we are
                # testing the infrastructure.
                # TODO(boomanaiden154): Remove this once the premerge pipeline is no
                # longer in a testing state and we can directly assert the workflow
                # result.
                for step in job.steps:
                    if step.conclusion != "success" and step.conclusion != "skipped":
                        job_result = 0
                        break

            queue_time = started_at - created_at
            run_time = completed_at - started_at

            if run_time.seconds == 0:
                continue

            # The timestamp associated with the event is expected by Grafana to be
            # in nanoseconds.
            completed_at_ns = int(completed_at.timestamp()) * 10**9

            logging.info(f"Adding a job metric for job {job.id} in workflow {task.id}")

            workflow_metrics.append(
                JobMetrics(
                    name_prefix + "_" + name_suffix,
                    queue_time.seconds,
                    run_time.seconds,
                    job_result,
                    completed_at_ns,
                    workflow_run.id,
                    workflow_run.name,
                )
            )

    return workflow_metrics, most_recent_workflow_processed


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
                f"{name} queue_time={workflow_metric.queue_time},run_time={workflow_metric.run_time},status={workflow_metric.status} {workflow_metric.created_at_ns}"
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
    github_last_seen_workflow = None

    # Enter the main loop. Every five minutes we wake up and dump metrics for
    # the relevant jobs.
    while True:
        github_object = Github(auth=github_auth)
        github_repo = github_object.get_repo("llvm/llvm-project")

        github_metrics, github_last_seen_workflow = get_per_workflow_metrics(
            github_repo, github_last_seen_workflow
        )
        sampled_metrics = get_sampled_workflow_metrics(github_repo)
        metrics = github_metrics + sampled_metrics

        upload_metrics(metrics, grafana_metrics_userid, grafana_api_key)
        logging.info(f"Uploaded {len(metrics)} metrics")

        time.sleep(SCRAPE_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
