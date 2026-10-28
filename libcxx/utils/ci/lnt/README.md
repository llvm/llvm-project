# LNT tools for libc++ performance tracking

This directory contains utilities for continuous benchmarking of libc++ with LNT.

## Gathering historical performance data

When generating historical performance data, benchmarking every commit of libc++
is prohibitively expensive. Furthermore, generating this data from scratch is
expected  to be common, since it must happen whenever a fixed parameter like the
compiler or the OS changes.

Instead, the tools in this directory aim to make it possible to generate historical
performance data quickly with coarse granularity, and to then gather finer granularity
data on commits that show a change.

Coarse grained performance data is obtained by determining "anchor commits", which
are libc++ commits that fall at specific intervals (e.g. the first commit of every
week). These anchor commits remain stable through time: they don't change as new
commits are introduced. This makes it possible to re-generate performance data for
the same anchor commits with a different configuration, and to compare across
configurations.

Finer granularity data is obtained by finding changepoints in the data, and then
requesting the benchmarking of additional commits based on the changes found.

The system is meant to be run periodically on a schedule. It keeps no state: on every
invocation, what should be measured is recomputed from Git and what has been measured
is recomputed from LNT and from the Github Actions API. This makes the overall system
converge towards a state where all the desired commits have been benchmarked, without
necessarily ever reaching it (as new desired commits are added).

To achieve this, several tools work together:

```
# What should have benchmark data: one commit per week since 2023.
select-anchor-commits --since 2023-01-02 --every week > anchors.txt

# Find changepoints in the existing data
curl -sS --fail --compressed -G https://lnt.llvm.org/api/suites/libcxx/series                                    \
     --data-urlencode machine=<machine> --data-urlencode metric=execution_time --data-urlencode has_ordinal=true \
  | detect-changepoints --series-by test --aggregate median --output changes.jsonl

# Determine which commits to refine based on the change points
curl "https://lnt.llvm.org/api/suites/libcxx/runs?machine=<machine>&limit=10000" | jq -r '.items[].commit' > attempted.txt
select-refinement-commits --changes changes.jsonl --attempted-commits attempted.txt \
                          --direction increase --output refinement.txt

# What is missing from LNT (we want at least 3 samples for each commit).
cat anchors.txt refinement.txt | plan-benchmarks --lnt-url https://lnt.llvm.org --test-suite libcxx \
                                                 --machine <machine> --samples 3 > plan.jsonl

# Request the corresponding workflow runs, at most 4 at a time to be a good citizen.
export GITHUB_TOKEN=$(gh auth token)
dispatch-benchmarks --work-items plan.jsonl --max-in-flight 4 --dry-run
```

In a nutshell, `select-anchor-commits` produces the list of anchor commits that we want
data for. `detect-changepoints` determines the changepoints that can be detected in the
current data, and `select-refinement-commits` ranks those changepoints and determines
which additional commits should be requested to refine the data (basically bisect highest
ranked change points).

`plan-benchmarks` then looks at which commits we actually already have data for in the LNT
instance and produces a plan of what runs we need to trigger in order to get data for the
missing commits, taking into account the number of samples we want for each commit. Finally,
`dispatch-benchmarks` interprets this plan and actually dispatches the Github workflows based
on a budget, taking into account currently running workflows and previously failed runs, if
any (to avoid requesting runs that fail indefinitely).

Note that since `dispatch-benchmarks` both reads and creates workflow runs, it requires a
GitHub token with write access to the repository. That token can either be passed as an
argument or picked up from the `GITHUB_TOKEN` environment variable.

In production, this pipeline is run by the `libcxx-benchmark-cron.yml` workflow, which runs
it for each machine defined in `machines.json` on a schedule.

## Configuring the benchmark machines

`machines.json` describes the machines we benchmark on and their configuration. It is the
single source of truth for all workflows that run benchmarks (PR benchmarking, running
historical benchmarks, etc). Each entry contains variables used by the various workflows
and the LNT machine name that the results will be reported under.

The `setup` key describes how benchmarks are run on that machine. It contains the following
entries:

- `runner`: the GitHub Actions runner(s) to benchmark on.
- `container`: the image to use if running in a container. Otherwise, the benchmarks run directly
               on the host.
- `cxx`: the C++ compiler to use.
- `xcode-version`: the version of Xcode to use (only relevant on macOS).
- `benchmark-suite-version`: the version of the benchmark suite to use (a LLVM monorepo SHA).
- `cmake-cache`: the CMake cache to use when building libc++ before running the benchmarks (only
                 relevant when benchmarking libc++).
- `test-config`: the Lit testing configuration used to run the benchmarks against that build (only
                 relevant when benchmarking libc++).
- `lit-params`: additional Lit parameters to pass when running the benchmarks.

The `coverage` key establishes how far back and at which frequency performance should be measured
for that configuration. A machine without a `coverage` entry can be defined, but it won't result
in historical data. Within it, the optional `refinement` key enables bisecting the changes found in
that data: it gives the `metric` to use for finding changepoints, the `direction` of the changes to
bisect (`increase`, `decrease` or `both`) and the number of commits (`max-commits`) to select at a time.

The `pr-benchmarks` key determines whether the machine is used when benchmarking pull requests
with `/libcxx-bot benchmark`.

## Running benchmarks locally

On GitHub, the `libcxx-lnt-pipeline.yml` workflow is used to run benchmarks and report
results to a LNT instance. This workflow wraps the `libcxx/utils/ci/lnt/run-benchmarks` script,
which can be used to benchmark locally:

```
run-benchmarks --test-suite-commit <SHA1> --machine <MACHINE>                 \
               --compiler clang++ --benchmark-commit <SHA2>                   \
               --test-config installed-libc++.cfg.in                          \
               --output result.json                                           \
               -- --param std=c++26 --param optimization=speed                \
                  --param libcxx_installation=<PATH>
```

This will run the benchmarks (using the test suite at the specified `SHA1`) against the installation
of libc++ at `PATH` (which is assumed to be libc++ as-of `SHA2`), and produce a LNT-ready JSON report.
The results can then be submitted to a LNT instance if desired (requires an API key). This is done with
the `lnt v5 submit` command from the `llvm-lnt` Python package, which reads one run per line:

```
pip install llvm-lnt
jq -c . result.json | lnt v5 submit https://lnt.llvm.org --suite libcxx --token <TOKEN>
```

Note that `run-benchmarks` does not build anything: the library being benchmarked must have been built
or installed beforehand. How to pick up that library is determined by `--test-config` and any Lit parameters
passed.

## Test suite schemas

`schemas/` holds the schemas of the `libcxx` and `libstdcxx` test suites as we submit them to LNT.
This describes the various metrics and fields we submit to the instance.

For the `libcxx` test suite, the commits are libc++ commits and their ordinal (the ordering between
commits) is obtained by `git rev-list --count`. Commits not on the `main` branch (e.g. commits made
on PR branches) don't have an ordinal.

For the `libstdcxx` test suite, the commits are versions of libstdc++ (e.g. `16.2.0`), and their
ordinal is `major * 10000 + minor * 100 + patch`.
