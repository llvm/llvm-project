# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Script for getting explanations from the premerge advisor."""

import argparse
import platform
import sys
import json

# TODO(boomanaiden154): Remove the optional call once we can require Python
# 3.10.
from typing import Optional

import requests
import github
import github.PullRequest

import generate_test_report_lib

PREMERGE_ADVISOR_URL = (
    "http://premerge-advisor.premerge-advisor.svc.cluster.local:5000/explain"
)
COMMENT_TAG = "<!--PREMERGE ADVISOR COMMENT: {platform}-->"


def get_comment_id(platform: str, pr: github.PullRequest.PullRequest) -> Optional[int]:
    platform_comment_tag = COMMENT_TAG.format(platform=platform)
    for comment in pr.as_issue().get_comments():
        if platform_comment_tag in comment.body:
            return comment.id
    return None


def get_comment(
    github_token: str,
    pr_number: int,
    body: str,
) -> dict[str, str]:
    repo = github.Github(github_token).get_repo("llvm/llvm-project")
    pr = repo.get_issue(pr_number).as_pull_request()
    comment = {"body": body}
    comment_id = get_comment_id(platform.system(), pr)
    if comment_id:
        comment["id"] = comment_id
    return comment


def main(
    commit_sha: str,
    build_log_files: list[str],
    github_token: str,
    pr_number: int,
    return_code: int,
):
    """The main entrypoint for the script.

    This function parses failures from files, requests information from the
    premerge advisor, and may write a Github comment depending upon the output.
    There are four different scenarios:
    1. There has never been a previous failure and the job passes - We do not
       create a comment. We write out an empty file to the comment path so the
       issue-write workflow knows not to create anything.
    2. There has never been a previous failure and the job fails - We create a
       new comment containing the failure information and any possible premerge
       advisor findings.
    3. There has been a previous failure and the job passes - We update the
       existing comment by passing its ID and a passed message to the
       issue-write workflow.
    4. There has been a previous failure and the job fails - We update the
       existing comment in the same manner as above, but generate the comment
       as if we have a failure.

    Args:
      commit_sha: The base commit SHA for this PR run.
      build_log_files: The list of JUnit XML files and ninja logs.
      github_token: The token to use to access the Github API.
      pr_number: The number of the PR associated with this run.
      return_code: The numerical return code of ninja/CMake.
    """
    if return_code == 0:
        with open("comment", "w") as comment_file_handle:
            comment = get_comment(
                github_token,
                pr_number,
                ":white_check_mark: With the latest revision this PR passed "
                "the premerge checks.",
            )
            if "id" in comment:
                json.dump([comment], comment_file_handle)
    junit_objects, ninja_logs = generate_test_report_lib.load_info_from_files(
        build_log_files
    )
    test_failures = generate_test_report_lib.get_failures(junit_objects)
    current_platform = f"{platform.system()}-{platform.machine()}".lower()
    explanation_request = {
        "base_commit_sha": commit_sha,
        "platform": current_platform,
        "failures": [],
    }
    if test_failures:
        for _, failures in test_failures.items():
            for name, failure_messsage in failures:
                explanation_request["failures"].append(
                    {"name": name, "message": failure_messsage}
                )
    else:
        ninja_failures = generate_test_report_lib.find_failure_in_ninja_logs(ninja_logs)
        for name, failure_message in ninja_failures:
            explanation_request["failures"].append(
                {"name": name, "message": failure_message}
            )
    advisor_response = requests.get(
        PREMERGE_ADVISOR_URL, json=explanation_request, timeout=5
    )
    if advisor_response.status_code == 200:
        print(advisor_response.json())
        comments = [
            get_comment(
                github_token,
                pr_number,
                generate_test_report_lib.generate_report(
                    generate_test_report_lib.compute_platform_title(),
                    return_code,
                    junit_objects,
                    ninja_logs,
                    failure_explanations_list=advisor_response.json(),
                ),
            )
        ]
        with open("comment", "w") as comment_file_handle:
            json.dump(comments, comment_file_handle)
    else:
        print(advisor_response.reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("commit_sha", help="The base commit SHA for the test.")
    parser.add_argument("return_code", help="The build's return code", type=int)
    parser.add_argument("github_token", help="Github authentication token", type=str)
    parser.add_argument("pr_number", help="The PR number", type=int)
    parser.add_argument(
        "build_log_files", help="Paths to JUnit report files and ninja logs.", nargs="*"
    )
    args = parser.parse_args()

    # Skip looking for results on AArch64 for now because the premerge advisor
    # service is not available on AWS currently.
    if platform.machine() == "arm64":
        sys.exit(0)

    main(
        args.commit_sha,
        args.build_log_files,
        args.github_token,
        args.pr_number,
        args.return_code,
    )
