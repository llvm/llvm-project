When PR is ready for merge, trigger a *pre merge* pipeline with a comment.
The comment set the toolchain version bump level - `[ci-merge-<bump_level>]`:
1. `[ci-merge-major]` - trigger *pre merge* pipeline that bump toolchaion MAJOR version (`X`).
2. `[ci-merge-minor]` - trigger *pre merge* pipeline that bump toolchaion MINOR version (`Y`).
3. `[ci-merge-patch]` - trigger *pre merge* pipeline that bump toolchaion PATCHLEVEL version (`Z`).
Toolchain versioning levels - `MAJOR.MINOR.PATCHLEVEL` (`X.Y.Z`).

For more details, review (pull request merge flow)[https://github.com/nextsilicon/next-llvm-project/tree/master?tab=readme-ov-file#pull-request-merge-flow]
