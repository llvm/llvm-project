import subprocess
import sys
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
        user_or_revert = "users/" in branch_name or "revert-" in branch_name
        origin_branch = branch_name.startswith("remotes/origin/")
        return user_or_revert and origin_branch

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


def generate_patch_for_branch(branch_name: str) -> bytes:
    command_vector = ["git", "diff", f"origin/main...origin/{branch_name}"]
    try:
        result = subprocess.run(
            command_vector, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as process_error:
        print(process_error.stderr)
        print(process_error.stdout)
        raise process_error
    return result.stdout


def generate_patches_for_all_branches(branches_to_remove: list[str], patches_path: str):
    for index, branch in enumerate(branches_to_remove):
        patch = generate_patch_for_branch(branch)
        patch_filename = branch.replace("/", "-") + ".patch"
        patch_path = os.path.join(patches_path, patch_filename)
        with open(patch_path, "wb") as patches_file_handle:
            patches_file_handle.write(patch)
        if index % 50 == 0:
            print(
                f"Finished generating patches for {index}/{len(branches_to_remove)} branches."
            )


def delete_branches(branches_to_remove: list[str]):
    for branch in branches_to_remove:
        # TODO(boomanaiden154): Only delete my branches for now to verify that
        # everything is working in the production environment.
        if "boomanaiden154" not in branch:
            continue
        command_vector = ["git", "push", "-d", "origin", branch]
        try:
            subprocess.run(
                command_vector,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as process_error:
            print(process_error.stderr)
            print(process_error.stdout)
        print(f"Deleted branch {branch}")


def main(github_token):
    if len(sys.argv) != 2:
        print(
            "Invalid invocation. Correct usage: python3 prune-unused-branches.py <patch output diectory>"
        )
        sys.exit(1)

    user_branches = get_branches()
    user_branches_from_prs = get_branches_from_open_prs(github_token)
    print(f"Found {len(user_branches)} user branches in the repository")
    print(f"Found {len(user_branches_from_prs)} user branches associated with PRs")
    user_branches_to_remove = get_user_branches_to_remove(
        user_branches, user_branches_from_prs
    )
    print(f"Deleting {len(user_branches_to_remove)} user branches.")
    generate_patches_for_all_branches(user_branches_to_remove, sys.argv[1])
    delete_branches(user_branches_to_remove)


if __name__ == "__main__":
    main(os.environ["GITHUB_TOKEN"])
