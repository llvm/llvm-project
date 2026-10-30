# LNT data for the LLVM test-suite

This directory configures the workflows that build Clang at a given commit, run the
[LLVM test-suite](https://github.com/llvm/llvm-test-suite) with it, and produce a report
for [LNT](https://lnt.llvm.org). Note that other submitting bots may exist outside of
this setup.

The `llvm-test-suite-lnt-pipeline.yml` workflow can be dispatched manually or through the
GitHub API with a commit and the name of a machine defined in `machines.json`. It uses the
`llvm-test-suite-run.yml` reusable workflow, which performs the actual build + run of the
LLVM Test Suite and uploads as an artifact.

The commit associated to the run data is the monorepo commit that Clang was built from.
Commits on `main` are given an ordinal (`git rev-list --count`) so they can be ordered
with each other, but other commits (e.g. PR commits) are reported without an ordinal.

## Configuring the machines

`machines.json` describes the machines we benchmark and their configuration. Each entry
contains the LNT machine name that the results are reported under, and a `setup` key
describing how the test-suite is run on that machine:

- `runner`: the GitHub Actions runner(s) to run on.
- `container`: the image to use if running in a container. Otherwise, everything runs directly
               on the host.
- `xcode-version`: the version of Xcode to use (only relevant on macOS).
- `llvm-cmake-cache`: the CMake cache used to build Clang, relative to the root of the monorepo.
- `test-suite-version`: the commit of the test-suite to use (a full SHA of llvm/llvm-test-suite).
- `run-test-suite-args`: additional arguments passed to `run-test-suite.py`, e.g. `-C O3`.

When changing any part of the setup, the name of the machine should be updated: the configuration
of a machine should be stable through time in order for time series to be meaningful.
