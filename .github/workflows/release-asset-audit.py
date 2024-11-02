import github
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


def main():
    token = sys.argv[1]

    gh = github.Github(login_or_token=token)
    repo = gh.get_repo("llvm/llvm-project")

    uploaders = set(
        [
            "DimitryAndric",
            "stefanp-ibm",
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

    for release in repo.get_releases():
        print("Release:", release.title)
        for asset in release.get_assets():
            created_at = asset.created_at
            updated_at = (
                "" if asset.created_at == asset.updated_at else asset.updated_at
            )
            print(
                f"{asset.name} : {asset.uploader.login} [{created_at} {updated_at}] ( {asset.download_count} )"
            )
            if not _is_valid(asset.uploader.login, uploaders, asset.name):
                with open('comment', 'w') as file:
                    file.write(f'@{asset.uploader.login} is not a valid uploader.')
                sys.exit(1)


if __name__ == "__main__":
    main()
