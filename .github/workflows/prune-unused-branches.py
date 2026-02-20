import subprocess
import sys
import os
import logging
import zipfile
import io
import urllib.request

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


def query_prs(github_token, extra_query_criteria) -> list[str]:
    gh = github.Github(auth=github.Auth.Token(github_token))
    query_template = """
    query ($after: String) {{
        search(query: "is:pr repo:llvm/llvm-project is:open {query_param}", type: ISSUE, first: 100, after: $after) {{
        nodes {{
            ... on PullRequest {{
            baseRefName
            headRefName
            isCrossRepository
            number
            }}
        }}
        pageInfo {{
            hasNextPage
            endCursor
        }}
        }}
    }}"""
    query = query_template.format(query_param=extra_query_criteria)
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

    return pr_data


def get_branches_from_open_prs(github_token) -> list[str]:
    pr_data = []
    pr_data.extend(query_prs(github_token, "head:users/"))
    # We need to explicitly check cases where the base is a user branch to
    # ensure we capture branches that are used as a diff base for cross-repo
    # PRs.
    pr_data.extend(query_prs(github_token, "base:users/"))

    user_branches = []
    for pr in pr_data:
        if not pr["isCrossRepository"]:
            if pr["baseRefName"] != "main":
                user_branches.append(pr["baseRefName"])
            user_branches.append(pr["headRefName"])
        else:
            # We want to skip cross-repo PRs where someone has simply used a
            # users/ branch naming scheme for a branch in their fork.
            if pr["baseRefName"] == "main":
                continue
            user_branches.append(pr["baseRefName"])
    # Convert to a set to ensure we have no duplicates.
    return list(set(user_branches))


def get_user_branches_to_remove(
    user_branches: list[str],
    user_branches_from_prs: list[str],
    previous_run_branches: list[str],
) -> list[str]:
    user_branches_to_remove = set(user_branches)
    for pr_user_branch in set(user_branches_from_prs):
        if pr_user_branch not in user_branches_to_remove:
            logging.warning(
                f"Found branch {pr_user_branch} attached to a PR, but it "
                "was not found in the repository. This is likely because "
                "the PR was created after this workflow cloned the repository."
            )
            continue
        user_branches_to_remove.remove(pr_user_branch)
    for branch in list(user_branches_to_remove):
        if branch not in previous_run_branches:
            user_branches_to_remove.remove(branch)
    return list(user_branches_to_remove)


def generate_patch_for_branch(branch_name: str) -> bytes:
    command_vector = [
        "git",
        "format-patch",
        "--stdout",
        "-k",
        f"origin/main..origin/{branch_name}",
    ]
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


def get_branches_found_in_previous_run(github_token: str) -> list[str]:
    gh = github.Github(auth=github.Auth.Token(github_token))
    repo = gh.get_repo("llvm/llvm-project")
    workflow_run = None
    for workflow_run in iter(
        repo.get_workflow("prune-branches.yml").get_runs(branch="main")
    ):
        if workflow_run.status == "completed":
            break
    assert workflow_run
    workflow_artifact = None
    for workflow_artifact in iter(workflow_run.get_artifacts()):
        if workflow_artifact.name == "BranchList":
            break
    assert workflow_artifact
    status, headers, response = workflow_artifact._requester.requestJson(
        "GET", workflow_artifact.archive_download_url
    )
    # Github will always send a redirect to where the file actuall lives.
    assert status == 302
    with urllib.request.urlopen(headers["location"]) as response:
        raw_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zip_file:
        branch_names = zip_file.read("branches.txt").decode("utf-8").split("\n")[:-1]
        return branch_names


def main(github_token):
    if len(sys.argv) != 2:
        print(
            "Invalid invocation. Correct usage: python3 prune-unused-branches.py <output diectory>"
        )
        sys.exit(1)

    previous_run_branches = get_branches_found_in_previous_run(github_token)
    print(
        f"{len(previous_run_branches)} branches existed the last time the workflow ran."
    )
    user_branches = get_branches()
    output_dir = sys.argv[1]
    with open(os.path.join(output_dir, "branches.txt"), "w") as branches_file:
        branches_file.writelines([user_branch + "\n" for user_branch in user_branches])
    user_branches_from_prs = get_branches_from_open_prs(github_token)
    print(f"Found {len(user_branches)} user branches in the repository")
    print(f"Found {len(user_branches_from_prs)} user branches associated with PRs")
    user_branches_to_remove = get_user_branches_to_remove(
        user_branches, user_branches_from_prs, previous_run_branches
    )
    print(f"Deleting {len(user_branches_to_remove)} user branches.")
    generate_patches_for_all_branches(
        user_branches_to_remove, os.path.join(output_dir, "patches")
    )
    delete_branches(user_branches_to_remove)


if __name__ == "__main__":
    main(os.environ["GITHUB_TOKEN"])
