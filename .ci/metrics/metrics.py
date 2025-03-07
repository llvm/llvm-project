import requests
import dateutil
import json
import time
import os
from dataclasses import dataclass
import sys
import collections
import logging

import github
from github import Github
from github import Auth

GRAFANA_URL = (
    "https://influx-prod-13-prod-us-east-0.grafana.net/api/v1/push/influx/write"
)
SCRAPE_INTERVAL_SECONDS = 5 * 60

# Number of builds to fetch per page. Since we scrape regularly, this can
# remain small.
BUILDKITE_GRAPHQL_BUILDS_PER_PAGE = 10

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
    """Fetches a page of the build list using the GraphQL BuildKite API.
    Returns the BUILDKITE_GRAPHQL_BUILDS_PER_PAGE last **finished** builds by
    default, or the BUILDKITE_GRAPHQL_BUILDS_PER_PAGE **finished** builds
    older than the one pointer by |cursor| if provided.
    The |cursor| value is taken from the previous page returned by the API.

    The returned data had the following format:

    Args:
      buildkite_token: the secret token to authenticate GraphQL requests.
      after_cursor: cursor after which to start the page fetch.

    Returns:
      The most recent builds after cursor (if set) with the following format:
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
    Note: for unknown reasons, graphql returns no jobs for a given build,
    while this endpoint does, hence why this uses this API instead of graphql.

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
        match_index = None
        for index, item in enumerate(page):
            if item["cursor"] == last_cursor:
                match_index = index
                break

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
    for build in reversed(builds):
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
            finished_at_ns = int(finished_at.timestamp()) * 10**9
            workflow_id = build["number"]
            workflow_name = "Github pull requests"
            output.append(
                JobMetrics(
                    job_name,
                    queue_time,
                    run_time,
                    status,
                    finished_at_ns,
                    workflow_id,
                    workflow_name,
                )
            )

    return output, last_recorded_build


def github_job_name_to_metric_name(workflow_name, job_name):
    workflow_key = GITHUB_WORKFLOW_TO_TRACK[workflow_name]
    job_key = GITHUB_JOB_TO_TRACK[workflow_key][job_name]
    return f"{workflow_key}_{job_key}"


def github_count_queued_running_workflows(workflow_list):
    """Returns the per-job count of running & queued jobs in the passed
    workflow list.

    Args:
      workflow_list: an iterable of workflows.

    Returns:
      A tuple, (per-job-queue-size, per-job-running-count). The key
      is the pretty job name, and the value the count of jobs.
    """
    queued_count = collections.Counter()
    running_count = collections.Counter()

    for workflow in workflow_list:
        if workflow.name not in GITHUB_WORKFLOW_TO_TRACK:
            continue

        workflow_key = GITHUB_WORKFLOW_TO_TRACK[workflow.name]
        for job in workflow.jobs():
            if job.name not in GITHUB_JOB_TO_TRACK[workflow_key]:
                continue
            job_key = GITHUB_JOB_TO_TRACK[workflow_key][job.name]
            metric_name = f"{workflow_key}_{job_key}"

            if job.status == "queued":
                queued_count[metric_name] += 1
            elif job.status == "in_progress":
                running_count[metric_name] += 1
    return queued_count, running_count


def get_sampled_workflow_metrics(github_repo: github.Repository):
    """Gets global statistics about the Github workflow queue

    Args:
      github_repo: A github repo object to use to query the relevant information.

    Returns:
      Returns a list of GaugeMetric objects, containing the relevant metrics about
      the workflow
    """
    # Other states are available (pending, waiting, etc), but the meaning
    # is not documented (See #70540).
    # "queued" seems to be the info we want.
    queued_1, running_1 = github_count_queued_running_workflows(
        github_repo.get_workflow_runs(status="queued")
    )
    queued_2, running_2 = github_count_queued_running_workflows(
        github_repo.get_workflow_runs(status="in_progress")
    )

    workflow_metrics = []
    for key, value in (queued_1 + queued_2).items():
        workflow_metrics.append(
            GaugeMetric(f"workflow_queue_size_{key}", value, time.time_ns())
        )
    for key, value in (running_1 + running_2).items():
        workflow_metrics.append(
            GaugeMetric(f"running_workflow_count_{key}", value, time.time_ns())
        )

    # Always send a hearbeat metric so we can monitor is this container is
    # still able to log to Grafana.
    workflow_metrics.append(
        GaugeMetric("metrics_container_heartbeat", 1, time.time_ns())
    )
    return workflow_metrics


def get_per_workflow_metrics(github_repo: github.Repository, last_workflow_id: str):
    """Gets the metrics for specified Github workflows.

    This function loads the last workflows from GitHub up to
    `last_workflow_id` and logs their metrics if they are referenced in
    GITHUB_WORKFLOW_TO_TRACK.
    The function returns a list of metrics, and the most recent processed
    workflow.
    If `last_workflow_id` is None, no metrics are returned, and the last
    completed github workflow ID is returned. This is used once when the
    program starts.

    Args:
      github_repo: A github repo object to use to query the relevant information.
      last_workflow_id: the last workflow we checked.

    Returns:
      Returns a list of JobMetrics objects, containing the relevant metrics about
      the workflow.
    """
    workflow_metrics = []
    last_recorded_workflow = None
    for workflow_run in iter(github_repo.get_workflow_runs(status="completed")):
        # Record the first workflow of this list as the most recent one.
        if last_recorded_workflow is None:
            last_recorded_workflow = workflow_run.id

        # If we saw this workflow already, break. We also break if no
        # workflow has been seen, as this means the script just started.
        if last_workflow_id == workflow_run.id or last_workflow_id is None:
            break

        # This workflow is not interesting to us. Skipping.
        if workflow_run.name not in GITHUB_WORKFLOW_TO_TRACK:
            continue

        workflow_key = GITHUB_WORKFLOW_TO_TRACK[workflow_run.name]

        for workflow_job in workflow_run.jobs():
            # This job is not interesting, skipping.
            if workflow_job.name not in GITHUB_JOB_TO_TRACK[workflow_key]:
                continue

            created_at = workflow_job.created_at
            started_at = workflow_job.started_at
            completed_at = workflow_job.completed_at
            job_result = int(workflow_job.conclusion == "success")
            job_key = GITHUB_JOB_TO_TRACK[workflow_key][workflow_job.name]

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
                    workflow_key + "_" + job_key,
                    queue_time.seconds,
                    run_time.seconds,
                    job_result,
                    created_at_ns,
                    workflow_run.id,
                    workflow_run.name,
                )
            )

    return workflow_metrics, last_recorded_workflow

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

    # This script only records workflows/jobs/builds finished after it
    # started. So we need to keep track of the last known build.
    buildkite_last_cursor = None
    github_last_workflow_id = None

    # Enter the main loop. Every five minutes we wake up and dump metrics for
    # the relevant jobs.
    while True:
        github_object = Github(auth=auth)
        github_repo = github_object.get_repo("llvm/llvm-project")

        buildkite_metrics, buildkite_last_cursor = buildkite_get_metrics(
            buildkite_token, buildkite_last_cursor
        )
        github_metrics, github_last_workflow_id = get_per_workflow_metrics(
            github_repo, github_last_workflow_id
        )
        sampled_metrics = get_sampled_workflow_metrics(github_repo)

        metrics = buildkite_metrics + github_metrics + sampled_metrics
        upload_metrics(metrics, grafana_metrics_userid, grafana_api_key)
        logging.info(f"Uploaded {len(metrics)} metrics")

        time.sleep(SCRAPE_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
