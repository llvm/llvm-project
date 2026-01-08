import github
import re
import sys

_SPECIAL_CASE_BINARIES = {
    "keith": {"clang+llvm-18.1.8-arm64-apple-macos11.tar.xz"},
}


def _is_valid(uploader_name, valid_uploaders, asset_name):
    if uploader_name in valid_uploaders:
        return True

    if uploader_name in _SPECIAL_CASE_BINARIES:
        return asset_name in _SPECIAL_CASE_BINARIES[uploader_name]

    return False


def _get_uploaders(release_version):
    # Until llvm 18, assets were uploaded by community members, the release managers
    # and the GitHub Actions bot.
    if release_version <= 18:
        return set(
            [
                "DimitryAndric",
                "stefanp-synopsys",
                "lei137",
                "omjavaid",
                "nicolerabjohn",
                "amy-kwan",
                "mandlebug",
                "zmodem",
                "androm3da",
                "tru",
                "rovka",
                "rorth",
                "quinnlp",
                "kamaub",
                "abrisco",
                "jakeegan",
                "maryammo",
                "tstellar",
                "github-actions[bot]",
            ]
        )
    # llvm 19 and beyond, only the release managers, bot and a much smaller
    # number of community members.
    elif release_version >= 19:
        return set(
            [
                "zmodem",
                "omjavaid",
                "tru",
                "tstellar",
                "github-actions[bot]",
                "c-rhodes",
                "dyung",
            ]
        )


def _get_major_release_version(release_title):
    # All release titles are of the form "LLVM X.Y.Z(-rcN)".
    match = re.match("LLVM ([0-9]+)\.", release_title)
    if match is None:
        _write_comment_and_exit_with_error(
            f'Could not parse release version from release title "{release_title}".'
        )
    else:
        return int(match.groups()[0])


def _write_comment_and_exit_with_error(comment):
    with open("comment", "w") as file:
        file.write(comment)
    sys.exit(1)


def main():
    token = sys.argv[1]

    gh = github.Github(login_or_token=token)
    repo = gh.get_repo("llvm/llvm-project")

    for release in repo.get_releases():
        print("Release:", release.title)
        uploaders = _get_uploaders(_get_major_release_version(release.title))
        for asset in release.get_assets():
            created_at = asset.created_at
            updated_at = (
                "" if asset.created_at == asset.updated_at else asset.updated_at
            )
            print(
                f"{asset.name} : {asset.uploader.login} [{created_at} {updated_at}] ( {asset.download_count} )"
            )
            if not _is_valid(asset.uploader.login, uploaders, asset.name):
                _write_comment_and_exit_with_error(
                    f"@{asset.uploader.login} is not a valid uploader."
                )


if __name__ == "__main__":
    main()
