# CI Improvement Proposals for cppa-clang

**Issue:** [CppDigest/llvm-project#1](https://github.com/CppDigest/llvm-project/issues/1)

---

## Executive Summary

8 proposals across 3 phases to reduce CI feedback from **120+ min → <30 min**.

| Phase | Timeline | Target | Key Actions |
|-------|----------|--------|-------------|
| Do Now | 1-2 weeks | < 60 min | sccache, path-filter, CMake cache |
| Do Next | 1-2 months | < 30 min | Flake quarantine, test sharding, incremental builds |
| Later | 3+ months | < 15 min | Self-hosted runners, agentic CI |

---

## Proposal Ranking

| # | Proposal | ROI | Effort | Risk | Access Required | Rollout | Phase |
|---|----------|-----|--------|------|-----------------|---------|-------|
| 1 | sccache optimization | High | Low | Low | Repo write | Edit premerge.yaml, test on fork, PR upstream | Do Now |
| 2 | Path-based selective CI | High | Med | Low | Repo write | Add path filters on fork, validate coverage, PR upstream | Do Now |
| 3 | CMake presets + config cache | Med | Low | Low | Repo write | Add preset file, test locally, PR upstream | Do Now |
| 4 | Flaky test quarantine | High | Med | Low | Repo write + CI logs | Build retry wrapper, monitor 2 weeks, PR upstream | Do Next |
| 5 | Parallel test sharding | Med | Med | Med | Repo write | Prototype on fork with matrix, measure, PR upstream | Do Next |
| 6 | Incremental builds on fork | High | Med | Med | Repo write + GH cache | Test cache key strategy on fork, measure hit rate | Do Next |
| 7 | Self-hosted runner fleet | High | High | Med | Org admin | Provision 1 runner, benchmark, scale gradually | Later |
| 8 | Agentic CI + web chat | High | High | Med | Org admin + webhooks | Deploy triage agent on fork, expand to bisection | Later |

At least 3 are directly implementable today: P1 (sccache), P2 (path-filter), P3 (CMake cache) — see Appendix A.

---

## Phased Roadmap

```
 Week 1-2 (Do Now)          Month 1-2 (Do Next)          Month 3+ (Later)
┌──────────────────┐       ┌──────────────────┐        ┌──────────────────┐
│ P1: sccache opt  │       │ P4: Flake quarnt │        │ P7: Self-hosted  │
│ P2: Path-filter  │──────>│ P5: Test shard   │───────>│ P8: Agent + chat │
│ P3: CMake cache  │       │ P6: Incr. build  │        │                  │
└──────────────────┘       └──────────────────┘        └──────────────────┘
 Target: <60 min            Target: <30 min              Target: <15 min
 Effort: 1-2 devs           Effort: 2-3 devs             Effort: team
```

## Metric Targets

| Metric | Baseline | After Do Now | After Do Next | After Later |
|--------|----------|----------|----------|----------|
| First signal time | ~120 min | < 60 min | < 30 min | < 15 min |
| sccache hit rate | Unknown | > 80% | > 85% | > 90% |
| Flake re-runs/day | ~5+ | ~5 | < 1 | < 1 |
| Build time (warm) | 120 min | 60 min | 30 min | 15 min |
| Agent-resolved | 0% | 0% | 0% | 60%+ |

---

## Appendix A: Proposal Details

### P1: sccache Optimization

| Aspect | Details |
|--------|---------|
| Problem | sccache is used but hit rate is untracked, likely suboptimal |
| Change | Pin version, set `SCCACHE_IDLE_TIMEOUT=0`, add hit-rate logging, warm cache on schedule |
| Location | `.github/workflows/premerge.yaml`, `.ci/` build scripts |
| Validation | Measure cache hit rate before/after; target >80% |

```yaml
env:
  SCCACHE_GHA_ENABLED: "true"
  SCCACHE_IDLE_TIMEOUT: "0"
  CMAKE_C_COMPILER_LAUNCHER: sccache
  CMAKE_CXX_COMPILER_LAUNCHER: sccache
steps:
  - uses: mozilla-actions/sccache-action@v0.0.6
  - name: Log sccache stats
    run: sccache --show-stats
    if: always()
```

### P2: Path-based Selective CI

| Aspect | Details |
|--------|---------|
| Problem | Full build on every PR regardless of changed files |
| Change | Add `paths`/`paths-ignore` filters to triggers |
| Location | `.github/workflows/premerge.yaml` trigger section |
| Validation | Track % of PRs skipping builds; no regressions in merged code |

```yaml
on:
  pull_request:
    paths: ['clang/**', 'llvm/**', 'cmake/**']
    paths-ignore: ['**/*.md', 'lldb/**', 'openmp/**']
```

### P3: CMake Presets + Config Cache

| Aspect | Details |
|--------|---------|
| Problem | CMake configure takes ~20 min (50% of clean-build time) |
| Change | `CMakePresets.json` with defaults; cache `CMakeCache.txt` between runs |
| Location | Root `CMakePresets.json`, `.ci/` scripts |
| Validation | Measure configure time before/after |

### P4: Flaky Test Quarantine

| Aspect | Details |
|--------|---------|
| Problem | lldb/openmp flaky tests cause daily spurious failures |
| Change | Auto-retry + quarantine list + flake dashboard |
| Location | `.ci/flaky-tests.txt`, test runner wrapper |
| Validation | Re-run rate before/after; target <1 spurious/day |

```
  Test fails ──→ Auto-retry (1x) ──→ Pass? ──→ Mark FLAKY (quarantine)
      │                                 No
      v                                 v
  Real failure                    Real failure
```

### P5: Parallel Test Sharding

| Aspect | Details |
|--------|---------|
| Problem | Tests run sequentially on one runner |
| Change | Shard lit tests across 3-4 parallel jobs (`--num-shards`/`--run-shard`) |
| Location | `premerge.yaml` test job, matrix strategy |
| Validation | Test wall-clock before/after |

### P6: Incremental Builds on Fork

| Aspect | Details |
|--------|---------|
| Problem | Every PR builds from scratch |
| Change | Cache build artifacts keyed by base branch SHA; rebuild only changed targets |
| Location | `premerge.yaml`, GitHub Actions cache |
| Validation | Build time for typical 1-5 file PRs |

### P7: Self-hosted Runner Fleet

| Aspect | Details |
|--------|---------|
| Problem | GitHub-hosted runners: 2-4 cores, cold cache every run |
| Change | Deploy 32-64 core runners with persistent sccache |
| Location | Runner config, `runs-on` labels |
| Validation | Wall-clock comparison + cost analysis |

| Spec | GitHub-hosted | Self-hosted |
|------|---------------|-------------|
| CPU | 2-4 cores | 32-64 cores |
| RAM | 7-14 GB | 64-128 GB |
| Disk | SSD (limited) | NVMe 1TB+ |
| Cache | Cold per-run | Persistent |
| Cost | $0.008/min | ~$0.002/min |

### P8: Agentic CI + Web Chat

| Aspect | Details |
|--------|---------|
| Problem | Failure triage, bisection, patching are all manual |
| Change | AI agents for classify → bisect → fix → submit; web chat for devs |
| Location | `.github/workflows/`, webhook integrations |
| Validation | Mean-time-to-resolution; developer satisfaction |

```
  CI Fail ──→ Classify ──→ Known flake? ──→ Auto-retry
                                No
                                v
              Bisect ──→ Propose Fix ──→ Submit PR
                                              │
                                              v
                                        Web Chat
                                        ("why did my PR fail?")
```

| Task | Automation | Human Role |
|------|-----------|------------|
| Failure classification | Full | Review |
| Flaky test detection | Full | Review quarantine |
| Regression bisection | Semi | Approve range |
| Fix proposal | Semi | Review + merge |
| Web chat Q&A | Full | Ask questions |

## Appendix B: Reference CI Config

```yaml
# .github/workflows/cppa-clang-ci.yml
name: cppa-clang CI
on:
  pull_request:
    paths: ['clang/**', 'llvm/**', 'cmake/**']
  push:
    branches: [main, 'release/**']

env:
  SCCACHE_GHA_ENABLED: "true"
  SCCACHE_IDLE_TIMEOUT: "0"

jobs:
  linux-build:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    steps:
      - uses: actions/checkout@v4
        with:
          sparse-checkout: |
            llvm
            clang
            cmake
            .ci
      - uses: mozilla-actions/sccache-action@v0.0.6
      - name: Configure
        run: |
          cmake -G Ninja -B build \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_ENABLE_PROJECTS="clang" \
            -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
            -DCMAKE_C_COMPILER_LAUNCHER=sccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
            -DLLVM_ENABLE_LLD=ON \
            llvm
      - name: Build
        run: ninja -C build -j$(nproc)
      - name: Test
        run: ninja -C build check-clang check-llvm
      - name: sccache stats
        run: sccache --show-stats
        if: always()

  windows-build:
    runs-on: windows-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
        with:
          sparse-checkout: |
            llvm
            clang
            cmake
            .ci
      - uses: mozilla-actions/sccache-action@v0.0.6
      - name: Configure & Build & Test
        run: |
          cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
          ninja -C build -j%NUMBER_OF_PROCESSORS%
          ninja -C build check-clang
        shell: cmd
```
