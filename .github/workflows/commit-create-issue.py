import github
import sys

token = sys.argv[1]
gh = github.Github(auth=github.Auth.Token(token))
repo = gh.get_repo("llvm/llvm-project")

length = "4 weeks"
names = " ".join(["@" + name.rstrip() for name in sys.stdin])


body = f"""### TLDR: If you want to retain your commit access, please comment on this issue.  Otherwise, you can unsubscribe from this issue and ignore it.  Commit access is not required to contribute to the project.  You can still create Pull Requests without commit access.

{names}

LLVM has a policy of downgrading write access to its repositories for accounts with long term inactivity. This is done because inactive accounts with high levels of access tend to be at increased risk of compromise and this is one tactic that the project employs to guard itself from malicious actors.  Note that write access is not required to contribute to the project.  You can still submit pull requests and have someone else merge them.

Our project policy is to ping anyone with less than five 'interactions' with the repositories over a 12 month period to see if they still need commit access.  An 'interaction' and be any one of:

* Pushing a commit.
* Merging a pull request (either their own or someone else’s).
* Commenting on a PR.

If you want to retain your commit access, please post a comment on this issue.  If you do not want to keep your commit access, you can just ignore this issue.  If you have not responded in {length}, then you will move moved from the 'write' role within the project to the 'triage' role.  The 'triage' role is still a privileged role and will allow you to do the following:

* Review Pull Requests.
* Comment on issues.
* Apply/dismiss labels.
* Close, reopen, and assign all issues and pull requests.
* Apply milestones.
* Mark duplicate issues and pull requests.
* Request pull request reviews.
* Hide anyone’s comments.

In the future, if you want to re-apply for commit access, you can follow the instructions
[here](https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access).
"""

issue = repo.create_issue(
    title="Do you still need commit access?", body=body, labels=["infra:commit-access"]
)
print(issue.html_url)
