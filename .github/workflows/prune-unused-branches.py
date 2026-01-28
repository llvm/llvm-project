import subprocess
import os

import github


def get_branches() -> list[str]:
    git_process = subprocess.run(
        ["git", "branch", "--all"], stdout=subprocess.PIPE, check=True
    )
    branches = [
        branch.strip() for branch in git_process.stdout.decode("utf-8").split("\n")
    ]

    def branch_filter(branch_name):
        return "users/" in branch_name or "revert-" in branch_name

    filtered_branches = list(filter(branch_filter, branches))
    return [branch.replace("remotes/origin/", "") for branch in filtered_branches]


def get_branches_from_open_prs(github_token) -> list[str]:
    gh = github.Github(auth=github.Auth.Token(github_token))
    query = """
  query ($after: String) {
    search(query: "is:pr repo:llvm/llvm-project is:open head:users/", type: ISSUE, first: 100, after: $after) {
      nodes {
        ... on PullRequest {
          baseRefName
          headRefName
          isCrossRepository
          number
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }"""
    pr_data = []
    has_next_page = True
    variables = {"after": None}
    while has_next_page:
        _, res_data = gh._Github__requester.graphql_query(query, variables=variables)
        page_info = res_data["data"]["search"]["pageInfo"]
        has_next_page = page_info["hasNextPage"]
        if has_next_page:
            variables["after"] = page_info["endCursor"]
        prs = res_data["data"]["search"]["nodes"]
        pr_data.extend(prs)
        print(f"Processed {len(prs)} PRs")

    user_branches = []
    for pr in pr_data:
        if not pr["isCrossRepository"]:
            if pr["baseRefName"] != "main":
                user_branches.append(pr["baseRefName"])
            user_branches.append(pr["headRefName"])
    return user_branches


def get_user_branches_to_remove(
    user_branches: list[str], user_branches_from_prs: list[str]
) -> list[str]:
    user_branches_to_remove = set(user_branches)
    for pr_user_branch in set(user_branches_from_prs):
        user_branches_to_remove.remove(pr_user_branch)
    return list(user_branches_to_remove)


def main(github_token):
    user_branches = get_branches()
    user_branches_from_prs = get_branches_from_open_prs(github_token)
    print(f"Found {len(user_branches)} user branches in the repository")
    print(f"Found {len(user_branches_from_prs)} user branches associated with PRs")
    user_branches_to_remove = get_user_branches_to_remove(
        user_branches, user_branches_from_prs
    )
    print(f"Deleting {len(user_branches_to_remove)} user branches.")
    print(user_branches_to_remove)


if __name__ == "__main__":
    main(os.environ["GITHUB_TOKEN"])
