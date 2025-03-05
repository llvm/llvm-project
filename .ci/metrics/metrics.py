import requests
import dateutil
import json
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
GITHUB_PROJECT = "llvm/llvm-project"
WORKFLOWS_TO_TRACK = ["LLVM Premerge Checks"]
SCRAPE_INTERVAL_SECONDS = 5 * 60

# Number of builds to fetch per page. Since we scrape regularly, this can
# remain small.
BUILDKITE_GRAPHQL_BUILDS_PER_PAGE = 10

# Lists the BuildKite jobs we want to track. Maps the BuildKite job name to
# the metric name in Grafana. This is important not to lose metrics history
# if the workflow name changes.
BUILDKITE_WORKFLOW_TO_TRACK = {
    ":linux: Linux x64": "buildkite_linux",
    ":windows: Windows x64": "buildkite_windows",
}

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


def buildkite_fetch_page_build_list(
    buildkite_token: str, after_cursor: str = None
) -> list[dict[str, str]]:
    """Fetches a page of the build list using the GraphQL BuildKite API. Returns the BUILDKITE_GRAPHQL_BUILDS_PER_PAGE last **finished** builds by default, or the BUILDKITE_GRAPHQL_BUILDS_PER_PAGE **finished** builds older than the one pointer by |cursor| if provided.
    The |cursor| value is taken from the previous page returned by the API.

    The returned data had the following format:

    Args:
      buildkite_token: the secret token to authenticate GraphQL requests.
      after_cursor: cursor after which to start the page fetch.

    Returns:
      Returns most recents builds after cursor (if set) with the following format:
      [
        {
            "cursor": <value>,
            "number": <build-number>,
        }
      ]
    """

    BUILDKITE_GRAPHQL_QUERY = """
  query OrganizationShowQuery {{
    organization(slug: "llvm-project") {{
      pipelines(search: "Github pull requests", first: 1) {{
        edges {{
          node {{
            builds (state: [FAILED, PASSED], first: {PAGE_SIZE}, after: {AFTER}) {{
              edges {{
                cursor
                node {{
                  number
                }}
              }}
            }}
          }}
        }}
      }}
    }}
  }}
  """
    data = BUILDKITE_GRAPHQL_QUERY.format(
        PAGE_SIZE=BUILDKITE_GRAPHQL_BUILDS_PER_PAGE,
        AFTER="null" if after_cursor is None else '"{}"'.format(after_cursor),
    )
    data = data.replace("\n", "").replace('"', '\\"')
    data = '{ "query": "' + data + '" }'
    url = "https://graphql.buildkite.com/v1"
    headers = {
        "Authorization": "Bearer " + buildkite_token,
        "Content-Type": "application/json",
    }
    r = requests.post(url, data=data, headers=headers)
    data = r.json()
    # De-nest the build list.
    builds = data["data"]["organization"]["pipelines"]["edges"][0]["node"]["builds"][
        "edges"
    ]
    # Fold cursor info into the node dictionnary.
    return [{**x["node"], "cursor": x["cursor"]} for x in builds]


def buildkite_get_build_info(build_number: str) -> dict:
    """Returns all the info associated with the provided build number.
    Note: for unknown reasons, graphql returns no jobs for a given build, while this endpoint does, hence why this uses this API instead of graphql.

      Args:
        build_number: which build number to fetch info for.

      Returns:
        The info for the target build, a JSON dictionnary.
    """

    URL = "https://buildkite.com/llvm-project/github-pull-requests/builds/{}.json"
    return requests.get(URL.format(build_number)).json()


def buildkite_get_builds_up_to(buildkite_token: str, last_cursor: str = None) -> list:
    """Returns the last BUILDKITE_GRAPHQL_BUILDS_PER_PAGE builds by default, or
    until the build pointed by |last_cursor| is found.

    Args:
     buildkite_token: the secret token to authenticate GraphQL requests.
     last_cursor: the cursor to stop at if set. If None, a full page is fetched.
    """
    output = []
    cursor = None

    while True:
        page = buildkite_fetch_page_build_list(buildkite_token, cursor)
        # No cursor provided, return the first page.
        if last_cursor is None:
            return page

        # Cursor has been provided, check if present in this page.
        match_index = next(
            (i for i, x in enumerate(page) if x["cursor"] == last_cursor), None
        )
        # Not present, continue loading more pages.
        if match_index is None:
            output += page
            cursor = page[-1]["cursor"]
            continue
        # Cursor found, keep results up to cursor
        output += page[:match_index]
        return output


def buildkite_get_metrics(
    buildkite_token: str, last_cursor: str = None
) -> (list[JobMetrics], str):
    """Returns a tuple with:
    - the metrics to record until |last_cursor| is reached, or none if last cursor is None.
    - the cursor of the most recent build processed.

    Args:
     buildkite_token: the secret token to authenticate GraphQL requests.
     last_cursor: the cursor to stop at if set. If None, a full page is fetched.
    """
    builds = buildkite_get_builds_up_to(buildkite_token, last_cursor)
    # Don't return any metrics if last_cursor is None.
    # This happens when the program starts.
    if last_cursor is None:
        return [], builds[0]["cursor"]

    last_recorded_build = last_cursor
    output = []
    for build in builds:
        info = buildkite_get_build_info(build["number"])
        last_recorded_build = build["cursor"]
        for job in info["jobs"]:
            # Skip this job.
            if job["name"] not in BUILDKITE_WORKFLOW_TO_TRACK:
                continue

            created_at = dateutil.parser.isoparse(job["created_at"])
            scheduled_at = dateutil.parser.isoparse(job["scheduled_at"])
            started_at = dateutil.parser.isoparse(job["started_at"])
            finished_at = dateutil.parser.isoparse(job["finished_at"])

            job_name = BUILDKITE_WORKFLOW_TO_TRACK[job["name"]]
            queue_time = (started_at - scheduled_at).seconds
            run_time = (finished_at - started_at).seconds
            status = bool(job["passed"])
            created_at_ns = int(created_at.timestamp()) * 10**9
            workflow_id = build["number"]
            workflow_name = "Github pull requests"
            output.append(
                JobMetrics(
                    job_name,
                    queue_time,
                    run_time,
                    status,
                    created_at_ns,
                    workflow_id,
                    workflow_name,
                )
            )

    return output, last_recorded_build


def get_sampled_workflow_metrics(github_repo: github.Repository):
    """Gets global statistics about the Github workflow queue

    Args:
      github_repo: A github repo object to use to query the relevant information.

    Returns:
      Returns a list of GaugeMetric objects, containing the relevant metrics about
      the workflow
    """
    queued_job_counts = {}
    running_job_counts = {}

    # Other states are available (pending, waiting, etc), but the meaning
    # is not documented (See #70540).
    # "queued" seems to be the info we want.
    for queued_workflow in github_repo.get_workflow_runs(status="queued"):
        if queued_workflow.name not in WORKFLOWS_TO_TRACK:
            continue
        for queued_workflow_job in queued_workflow.jobs():
            job_name = queued_workflow_job.name
            # Workflows marked as queued can potentially only have some jobs
            # queued, so make sure to also count jobs currently in progress.
            if queued_workflow_job.status == "queued":
                if job_name not in queued_job_counts:
                    queued_job_counts[job_name] = 1
                else:
                    queued_job_counts[job_name] += 1
            elif queued_workflow_job.status == "in_progress":
                if job_name not in running_job_counts:
                    running_job_counts[job_name] = 1
                else:
                    running_job_counts[job_name] += 1

    for running_workflow in github_repo.get_workflow_runs(status="in_progress"):
        if running_workflow.name not in WORKFLOWS_TO_TRACK:
            continue
        for running_workflow_job in running_workflow.jobs():
            job_name = running_workflow_job.name
            if running_workflow_job.status != "in_progress":
                continue

            if job_name not in running_job_counts:
                running_job_counts[job_name] = 1
            else:
                running_job_counts[job_name] += 1

    workflow_metrics = []
    for queued_job in queued_job_counts:
        workflow_metrics.append(
            GaugeMetric(
                f"workflow_queue_size_{queued_job}",
                queued_job_counts[queued_job],
                time.time_ns(),
            )
        )
    for running_job in running_job_counts:
        workflow_metrics.append(
            GaugeMetric(
                f"running_workflow_count_{running_job}",
                running_job_counts[running_job],
                time.time_ns(),
            )
        )
    # Always send a hearbeat metric so we can monitor is this container is still able to log to Grafana.
    workflow_metrics.append(
        GaugeMetric("metrics_container_heartbeat", 1, time.time_ns())
    )
    return workflow_metrics

def get_per_workflow_metrics(
    github_repo: github.Repository, workflows_to_track: dict[str, int]
):
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
    workflow_metrics = []

    workflows_to_include = set(workflows_to_track.keys())

    for workflow_run in iter(github_repo.get_workflow_runs()):
        if len(workflows_to_include) == 0:
            break

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

        for workflow_job in workflow_jobs:
            created_at = workflow_job.created_at
            started_at = workflow_job.started_at
            completed_at = workflow_job.completed_at

            job_result = int(workflow_job.conclusion == "success")
            if job_result:
                # We still might want to mark the job as a failure if one of the steps
                # failed. This is required due to use setting continue-on-error in
                # the premerge pipeline to prevent sending emails while we are
                # testing the infrastructure.
                # TODO(boomanaiden154): Remove this once the premerge pipeline is no
                # longer in a testing state and we can directly assert the workflow
                # result.
                for step in workflow_job.steps:
                    if step.conclusion != "success" and step.conclusion != "skipped":
                        job_result = 0
                        break

            queue_time = started_at - created_at
            run_time = completed_at - started_at

            if run_time.seconds == 0:
                continue

            # The timestamp associated with the event is expected by Grafana to be
            # in nanoseconds.
            created_at_ns = int(created_at.timestamp()) * 10**9

            logging.info(
                f"Adding a job metric for job {workflow_job.id} in workflow {workflow_run.id}"
            )

            workflow_metrics.append(
                JobMetrics(
                    workflow_run.name + "-" + workflow_job.name,
                    queue_time.seconds,
                    run_time.seconds,
                    job_result,
                    created_at_ns,
                    workflow_run.id,
                    workflow_run.name,
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
    auth = Auth.Token(os.environ["GITHUB_TOKEN"])
    grafana_api_key = os.environ["GRAFANA_API_KEY"]
    grafana_metrics_userid = os.environ["GRAFANA_METRICS_USERID"]
    buildkite_token = os.environ["BUILDKITE_TOKEN"]

    # The last buildkite build recorded.
    buildkite_last_cursor = None

    workflows_to_track = {}
    for workflow_to_track in WORKFLOWS_TO_TRACK:
        workflows_to_track[workflow_to_track] = None

    # Enter the main loop. Every five minutes we wake up and dump metrics for
    # the relevant jobs.
    while True:
        github_object = Github(auth=auth)
        github_repo = github_object.get_repo("llvm/llvm-project")

        current_metrics, buildkite_last_cursor = buildkite_get_metrics(
            buildkite_token, buildkite_last_cursor
        )
        current_metrics += get_per_workflow_metrics(github_repo, workflows_to_track)
        current_metrics += get_sampled_workflow_metrics(github_repo)

        upload_metrics(current_metrics, grafana_metrics_userid, grafana_api_key)
        logging.info(f"Uploaded {len(current_metrics)} metrics")

        for workflow_metric in reversed(current_metrics):
            if isinstance(workflow_metric, JobMetrics):
                workflows_to_track[
                    workflow_metric.workflow_name
                ] = workflow_metric.workflow_id

        time.sleep(SCRAPE_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
