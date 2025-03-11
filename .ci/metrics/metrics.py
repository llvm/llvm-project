import collections
import datetime
import github
import logging
import os
import requests
import sys
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

# The number of workflows to pull when sampling Github workflows.
# - Github API filtering is broken: we cannot apply any filtering:
# - See https://github.com/orgs/community/discussions/86766
# - A workflow can complete before another workflow, even when starting later.
# - We don't want to sample the same workflow twice.
#
# This means we essentially have a list of workflows sorted by creation date,
# and that's all we can deduce from it. So for each iteration, we'll blindly
# process the last N workflows.
GITHUB_WORKFLOWS_MAX_PROCESS_COUNT = 1000
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
    completed_at_ns: int
    workflow_id: int
    workflow_name: str

@dataclass
class GaugeMetric:
    name: str
    value: int
    time_ns: int


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

        if task.status == "completed":
            workflow_seen_as_completed.add(task.id)

        # This workflow has already been seen completed in the previous run.
        if task.id in last_workflows_seen_as_completed:
            continue

        name_prefix = GITHUB_WORKFLOW_TO_TRACK[task.name]
        for job in task.jobs():
            # This job is not interesting to us.
            if job.name not in GITHUB_JOB_TO_TRACK[name_prefix]:
                continue

            name_suffix = GITHUB_JOB_TO_TRACK[name_prefix][job.name]
            metric_name = name_prefix + "_" + name_suffix

            if task.status != "completed":
                if job.status == "queued":
                    queued_count[metric_name] += 1
                elif job.status == "in_progress":
                    running_count[metric_name] += 1
                continue

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

            created_at = job.created_at
            started_at = job.started_at
            completed_at = job.completed_at
            queue_time = started_at - created_at
            run_time = completed_at - started_at
            if run_time.seconds == 0:
                continue

            # Grafana will refuse to ingest metrics older than ~2 hours, so we
            # should avoid sending historical data.
            metric_age_mn = (
                datetime.datetime.now(datetime.timezone.utc) - completed_at
            ).total_seconds() / 60
            if metric_age_mn > GRAFANA_METRIC_MAX_AGE_MN:
                logging.info(
                    f"Job {job.id} from workflow {task.id} dropped due"
                    + f" to staleness: {metric_age_mn}mn old."
                )
                continue

            logging.info(f"Adding a job metric for job {job.id} in workflow {task.id}")
            # The timestamp associated with the event is expected by Grafana to be
            # in nanoseconds.
            completed_at_ns = int(completed_at.timestamp()) * 10**9
            workflow_metrics.append(
                JobMetrics(
                    metric_name,
                    queue_time.seconds,
                    run_time.seconds,
                    job_result,
                    completed_at_ns,
                    task.id,
                    task.name,
                )
            )

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

        metrics, gh_last_workflows_seen_as_completed = github_get_metrics(
            github_repo, gh_last_workflows_seen_as_completed
        )
        upload_metrics(metrics, grafana_metrics_userid, grafana_api_key)
        logging.info(f"Uploaded {len(metrics)} metrics")

        time.sleep(SCRAPE_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
