#!/usr/bin/env python3
# ===-- commit-access-review.py  --------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#
#
# ===------------------------------------------------------------------------===#

import datetime
import github
import re
import requests
import time
import sys
import re


class User:
    THRESHOLD = 5

    def __init__(self, name, triage_list):
        self.name = name
        self.authored = 0
        self.merged = 0
        self.reviewed = 0
        self.triage_list = triage_list

    def add_authored(self, val=1):
        self.authored += val
        if self.meets_threshold():
            print(self.name, "meets the threshold with authored commits")
            del self.triage_list[self.name]

    def set_authored(self, val):
        self.authored = 0
        self.add_authored(val)

    def add_merged(self, val=1):
        self.merged += val
        if self.meets_threshold():
            print(self.name, "meets the threshold with merged commits")
            del self.triage_list[self.name]

    def add_reviewed(self, val=1):
        self.reviewed += val
        if self.meets_threshold():
            print(self.name, "meets the threshold with reviewed commits")
            del self.triage_list[self.name]

    def get_total(self):
        return self.authored + self.merged + self.reviewed

    def meets_threshold(self):
        return self.get_total() >= self.THRESHOLD

    def __repr__(self):
        return "{} : a: {} m: {} r: {}".format(
            self.name, self.authored, self.merged, self.reviewed
        )


def check_manual_requests(
    gh: github.Github, start_date: datetime.datetime
) -> list[str]:
    """
    Return a list of users who have been asked since ``start_date`` if they
    want to keep their commit access or if they have applied for commit
    access since ``start_date``
    """

    query = """
        query ($query: String!, $after: String) {
          search(query: $query, type: ISSUE, first: 100, after: $after) {
            nodes {
              ... on Issue {
                author {
                  login
                }
                body
              }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
        """
    formatted_start_date = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    variables = {
        "query": f"type:issue created:>{formatted_start_date} org:llvm repo:llvm-project label:infra:commit-access,infra:commit-access-request"
    }

    has_next_page = True
    users = []
    while has_next_page:
        res_header, res_data = gh._Github__requester.graphql_query(
            query=query, variables=variables
        )
        data = res_data["data"]
        for issue in data["search"]["nodes"]:
            users.extend([user[1:] for user in re.findall("@[^ ,\n]+", issue["body"])])
            if issue["author"]:
                users.append(issue["author"]["login"])
        has_next_page = data["search"]["pageInfo"]["hasNextPage"]
        if has_next_page:
            variables["after"] = data["search"]["pageInfo"]["endCursor"]
    return users


def get_num_commits(gh: github.Github, user: str, start_date: datetime.datetime) -> int:
    """
    Get number of commits that ``user`` has been made since ``start_date`.
    """
    variables = {
        "owner": "llvm",
        "user": user,
        "start_date": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    user_query = """
        query ($user: String!) {
          user(login: $user) {
            id
          }
        }
    """

    res_header, res_data = gh._Github__requester.graphql_query(
        query=user_query, variables=variables
    )
    data = res_data["data"]
    variables["user_id"] = data["user"]["id"]

    query = """
        query ($owner: String!, $user_id: ID!, $start_date: GitTimestamp!){
          organization(login: $owner) {
            teams(query: "llvm-committers" first:1) {
              nodes {
                repositories {
                  nodes {
                    ref(qualifiedName: "main") {
                      target {
                        ... on Commit {
                          history(since: $start_date, author: {id: $user_id }) {
                            totalCount
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
     """
    count = 0
    res_header, res_data = gh._Github__requester.graphql_query(
        query=query, variables=variables
    )
    data = res_data["data"]
    for repo in data["organization"]["teams"]["nodes"][0]["repositories"]["nodes"]:
        count += int(repo["ref"]["target"]["history"]["totalCount"])
        if count >= User.THRESHOLD:
            break
    return count


def is_new_committer_query_repo(
    gh: github.Github, user: str, start_date: datetime.datetime
) -> bool:
    """
    Determine if ``user`` is a new committer.  A new committer can keep their
    commit access even if they don't meet the criteria.
    """
    variables = {
        "user": user,
    }

    user_query = """
        query ($user: String!) {
          user(login: $user) {
            id
          }
        }
    """

    res_header, res_data = gh._Github__requester.graphql_query(
        query=user_query, variables=variables
    )
    data = res_data["data"]
    variables["owner"] = "llvm"
    variables["user_id"] = data["user"]["id"]
    variables["start_date"] = start_date.strftime("%Y-%m-%dT%H:%M:%S")

    query = """
        query ($owner: String!, $user_id: ID!){
          organization(login: $owner) {
            repository(name: "llvm-project") {
              ref(qualifiedName: "main") {
                target {
                  ... on Commit {
                    history(author: {id: $user_id }, first: 5) {
                      nodes {
                        committedDate
                      }
                    }
                  }
                }
              }
            }
          }
        }
     """

    res_header, res_data = gh._Github__requester.graphql_query(
        query=query, variables=variables
    )
    data = res_data["data"]
    repo = data["organization"]["repository"]
    commits = repo["ref"]["target"]["history"]["nodes"]
    if len(commits) == 0:
        return True
    committed_date = commits[-1]["committedDate"]
    if datetime.datetime.strptime(committed_date, "%Y-%m-%dT%H:%M:%SZ") < start_date:
        return False
    return True


def is_new_committer(
    gh: github.Github, user: str, start_date: datetime.datetime
) -> bool:
    """
    Wrapper around is_new_commiter_query_repo to handle exceptions.
    """
    try:
        return is_new_committer_query_repo(gh, user, start_date)
    except:
        pass
    return True


def get_review_count(
    gh: github.Github, user: str, start_date: datetime.datetime
) -> int:
    """
    Return the number of reviews that ``user`` has done since ``start_date``.
    """
    query = """
        query ($query: String!) {
          search(query: $query, type: ISSUE, first: 5) {
            issueCount
          }
        }
        """
    formatted_start_date = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    variables = {
        "owner": "llvm",
        "repo": "llvm-project",
        "user": user,
        "query": f"type:pr commenter:{user} -author:{user} merged:>{formatted_start_date} org:llvm",
    }

    res_header, res_data = gh._Github__requester.graphql_query(
        query=query, variables=variables
    )
    data = res_data["data"]
    return int(data["search"]["issueCount"])


def count_prs(gh: github.Github, triage_list: dict, start_date: datetime.datetime):
    """
    Fetch all the merged PRs for the project since ``start_date`` and update
    ``triage_list`` with the number of PRs merged for each user.
    """

    query = """
        query ($query: String!, $after: String) {
          search(query: $query, type: ISSUE, first: 100, after: $after) {
            issueCount,
            nodes {
              ... on PullRequest {
                 author {
                   login
                 }
                 mergedBy {
                   login
                 }
              }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
    """
    date_begin = start_date
    date_end = None
    while date_begin < datetime.datetime.now():
        date_end = date_begin + datetime.timedelta(days=7)
        formatted_date_begin = date_begin.strftime("%Y-%m-%dT%H:%M:%S")
        formatted_date_end = date_end.strftime("%Y-%m-%dT%H:%M:%S")
        variables = {
            "query": f"type:pr is:merged merged:{formatted_date_begin}..{formatted_date_end} org:llvm",
        }
        has_next_page = True
        while has_next_page:
            print(variables)
            res_header, res_data = gh._Github__requester.graphql_query(
                query=query, variables=variables
            )
            data = res_data["data"]
            for pr in data["search"]["nodes"]:
                # Users can be None if the user has been deleted.
                if not pr["author"]:
                    continue
                author = pr["author"]["login"]
                if author in triage_list:
                    triage_list[author].add_authored()

                if not pr["mergedBy"]:
                    continue
                merger = pr["mergedBy"]["login"]
                if author == merger:
                    continue
                if merger not in triage_list:
                    continue
                triage_list[merger].add_merged()

            has_next_page = data["search"]["pageInfo"]["hasNextPage"]
            if has_next_page:
                variables["after"] = data["search"]["pageInfo"]["endCursor"]
        date_begin = date_end


def main():
    token = sys.argv[1]
    gh = github.Github(login_or_token=token)
    org = gh.get_organization("llvm")
    repo = org.get_repo("llvm-project")
    one_year_ago = datetime.datetime.now() - datetime.timedelta(days=365)
    triage_list = {}
    for collaborator in repo.get_collaborators(permission="push"):
        triage_list[collaborator.login] = User(collaborator.login, triage_list)

    print("Start:", len(triage_list), "triagers")
    # Step 0 Check if users have requested commit access in the last year.
    for user in check_manual_requests(gh, one_year_ago):
        if user in triage_list:
            print(user, "requested commit access in the last year.")
            del triage_list[user]
    print("After Request Check:", len(triage_list), "triagers")

    # Step 1 count all PRs authored or merged
    count_prs(gh, triage_list, one_year_ago)

    print("After PRs:", len(triage_list), "triagers")

    if len(triage_list) == 0:
        sys.exit(0)

    # Step 2 check for reviews
    for user in list(triage_list.keys()):
        review_count = get_review_count(gh, user, one_year_ago)
        triage_list[user].add_reviewed(review_count)

    print("After Reviews:", len(triage_list), "triagers")

    if len(triage_list) == 0:
        sys.exit(0)

    # Step 3 check for number of commits
    for user in list(triage_list.keys()):
        num_commits = get_num_commits(gh, user, one_year_ago)
        # Override the total number of commits to not double count commits and
        # authored PRs.
        triage_list[user].set_authored(num_commits)

    print("After Commits:", len(triage_list), "triagers")

    # Step 4 check for new committers
    for user in list(triage_list.keys()):
        print("Checking", user)
        if is_new_committer(gh, user, one_year_ago):
            print("Removing new committer: ", user)
            del triage_list[user]

    print("Complete:", len(triage_list), "triagers")

    with open("triagers.log", "w") as triagers_log:
        for user in triage_list:
            print(triage_list[user].__repr__())
            triagers_log.write(user + "\n")


if __name__ == "__main__":
    main()
