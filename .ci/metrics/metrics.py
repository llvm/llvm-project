import requests
import time
import os
from dataclasses import dataclass
import sys

import github
from github import Github
from github import Auth

GRAFANA_URL = (
    "https://influx-prod-13-prod-us-east-0.grafana.net/api/v1/push/influx/write"
)
GITHUB_PROJECT = "llvm/llvm-project"
WORKFLOWS_TO_TRACK = ["Check code formatting", "LLVM Premerge Checks"]
SCRAPE_INTERVAL_SECONDS = 5 * 60


@dataclass
class JobMetrics:
    job_name: str
    queue_time: int
    run_time: int
    status: int
    created_at_ns: int
    workflow_id: int


def get_metrics(github_repo: github.Repository, workflows_to_track: dict[str, int]):
    """Gets the metrics for specified Github workflows.

    This function takes in a list of workflows to track, and optionally the
    workflow ID of the last tracked invocation. It grabs the relevant data
    from Github, returning it to the caller.

    Args:
      github_repo: A github repo object to use to query the relevant information.
      workflows_to_track: A dictionary mapping workflow names to the last
        invocation ID where metrics have been collected, or None to collect the
        last five results.

    Returns:
      Returns a list of JobMetrics objects, containing the relevant metrics about
      the workflow.
    """
    workflow_runs = iter(github_repo.get_workflow_runs())

    workflow_metrics = []

    workflows_to_include = set(workflows_to_track.keys())

    while len(workflows_to_include) > 0:
        workflow_run = next(workflow_runs)
        if workflow_run.status != "completed":
            continue

        # This workflow was already sampled for this run, or is not tracked at
        # all. Ignoring.
        if workflow_run.name not in workflows_to_include:
            continue

        # There were no new workflow invocations since the previous scrape.
        # The API returns a sorted list with the most recent invocations first,
        # so we can stop looking for this particular workflow. Continue to grab
        # information on the other workflows of interest, if present.
        if workflows_to_track[workflow_run.name] == workflow_run.id:
            workflows_to_include.remove(workflow_run.name)
            continue

        workflow_jobs = workflow_run.jobs()
        if workflow_jobs.totalCount == 0:
            continue
        if workflow_jobs.totalCount > 1:
            raise ValueError(
                f"Encountered an unexpected number of jobs: {workflow_jobs.totalCount}"
            )

        created_at = workflow_jobs[0].created_at
        started_at = workflow_jobs[0].started_at
        completed_at = workflow_jobs[0].completed_at

        job_result = int(workflow_jobs[0].conclusion == "success")

        queue_time = started_at - created_at
        run_time = completed_at - started_at

        if run_time.seconds == 0:
            continue

        if (
            workflows_to_track[workflow_run.name] is None
            or workflows_to_track[workflow_run.name] == workflow_run.id
        ):
            workflows_to_include.remove(workflow_run.name)
        if (
            workflows_to_track[workflow_run.name] is not None
            and len(workflows_to_include) == 0
        ):
            break

        # The timestamp associated with the event is expected by Grafana to be
        # in nanoseconds.
        created_at_ns = int(created_at.timestamp()) * 10**9

        workflow_metrics.append(
            JobMetrics(
                workflow_run.name,
                queue_time.seconds,
                run_time.seconds,
                job_result,
                created_at_ns,
                workflow_run.id,
            )
        )

    return workflow_metrics


def upload_metrics(workflow_metrics, metrics_userid, api_key):
    """Upload metrics to Grafana.

    Takes in a list of workflow metrics and then uploads them to Grafana
    through a REST request.

    Args:
      workflow_metrics: A list of metrics to upload to Grafana.
      metrics_userid: The userid to use for the upload.
      api_key: The API key to use for the upload.
    """
    metrics_batch = []
    for workflow_metric in workflow_metrics:
        workflow_formatted_name = workflow_metric.job_name.lower().replace(" ", "_")
        metrics_batch.append(
            f"{workflow_formatted_name} queue_time={workflow_metric.queue_time},run_time={workflow_metric.run_time},status={workflow_metric.status} {workflow_metric.created_at_ns}"
        )

    request_data = "\n".join(metrics_batch)
    response = requests.post(
        GRAFANA_URL,
        headers={"Content-Type": "text/plain"},
        data=request_data,
        auth=(metrics_userid, api_key),
    )

    if response.status_code < 200 or response.status_code >= 300:
        print(
            f"Failed to submit data to Grafana: {response.status_code}", file=sys.stderr
        )


def main():
    # Authenticate with Github
    auth = Auth.Token(os.environ["GITHUB_TOKEN"])
    github_object = Github(auth=auth)
    github_repo = github_object.get_repo("llvm/llvm-project")

    grafana_api_key = os.environ["GRAFANA_API_KEY"]
    grafana_metrics_userid = os.environ["GRAFANA_METRICS_USERID"]

    workflows_to_track = {}
    for workflow_to_track in WORKFLOWS_TO_TRACK:
        workflows_to_track[workflow_to_track] = None

    # Enter the main loop. Every five minutes we wake up and dump metrics for
    # the relevant jobs.
    while True:
        current_metrics = get_metrics(github_repo, workflows_to_track)
        if len(current_metrics) == 0:
            print("No metrics found to upload.", file=sys.stderr)
            continue

        upload_metrics(current_metrics, grafana_metrics_userid, grafana_api_key)
        print(f"Uploaded {len(current_metrics)} metrics", file=sys.stderr)

        for workflow_metric in reversed(current_metrics):
            workflows_to_track[workflow_metric.job_name] = workflow_metric.workflow_id

        time.sleep(SCRAPE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
