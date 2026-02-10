# LLVM CI/Automation Workflow Report

**Issue:** [CppDigest/llvm-project#1](https://github.com/CppDigest/llvm-project/issues/1)

---

## Executive Summary

| Item | Value |
|------|-------|
| Pre-merge gate | `premerge.yaml` (GitHub Actions, every PR) |
| Build time | **120 min** (Linux) / **180 min** (Windows) |
| Post-commit bots | **200+** buildbots, never fully green |
| Flaky tests | lldb, openmp — spurious failures daily |
| Daily commits | 150+ on main |
| Legacy infra | `google/llvm-premerge-checks` archived June 2025 |
| Total workflow runs | 1.6M+ |
| Main bottleneck | Compilation (46% of pipeline time) |

---

## 1. PR Lifecycle

```
  PR opened
      │
      ├──→ [Label & Format]  ──→ pr-subscriber, labelling, clang-format  (~seconds)
      │
      ├──→ [Pre-merge Gate]  ──→ premerge.yaml: cmake → build → test    (~120-180 min)
      │         │
      │         └──→ Fail → Author fixes → re-push → re-run
      │
      ├──→ [Human Review]    ──→ Reviewer approval                      (hours-days)
      │
      └──→ [Merge]           ──→ Squash-merge to main
              │
              └──→ [Post-commit Bots]  ──→ 200+ buildbots (multi-arch)  (async)
                        │
                        └──→ Fail → Revert or fix-forward (manual)
```

## 2. Workflow Inventory

| Category | Count | Key Workflows | Trigger |
|----------|-------|--------------|---------|
| Core CI / Pre-merge | ~5 | `premerge.yaml`, `check-code-formatting`, `labelling` | `pull_request` |
| Subproject CI | ~15 | `libcxx-build-and-test.yaml`, `llvm-tests.yml` | `pull_request` (path-filtered) |
| Release | ~5 | `release-binaries.yml`, `release-asset-audit` | `workflow_dispatch` / `schedule` |
| Triage / Bots | ~8 | `pr-subscriber.yml`, `issue-comment` | `pull_request` / `issue_comment` |
| Maintenance | ~5 | Stale-issue cleanup, audit | `schedule` |
| Build helpers | ~10 | Container builds, cache warming | Reusable workflows |
| **Total** | **~48** | | |

## 3. Time Allocation (Pre-merge, Linux)

```
 Phase             Time     Share
 ─────────────────────────────────────────────────────
 Checkout + deps    5 min    4%   ██
 CMake configure   20 min   17%   ████████
 Compilation       55 min   46%   ██████████████████████████
 Linking           15 min   13%   ███████
 Testing (lit)     20 min   17%   ████████
 Artifacts          5 min    4%   ██
 ─────────────────────────────────────────────────────
 TOTAL            120 min  100%   Windows: +60 min overhead
```

| Phase | Linux | Windows | Bottleneck |
|-------|-------|---------|------------|
| Checkout | 5 min | 8 min | Monorepo ~3 GB |
| CMake | 20 min | 25 min | 50% of clean-build |
| Compile | 55 min | 80 min | 2.5M+ lines C++ |
| Link | 15 min | 25 min | Debug info OOM risk |
| Test | 20 min | 35 min | lldb/openmp flaky |
| **Total** | **120 min** | **180 min** | |

## 4. Flakiness & Signal Quality

| Source | Frequency | Impact |
|--------|-----------|--------|
| lldb tests | Daily | Spurious timeouts, race conditions |
| openmp / libomptarget | Daily | Thread-sensitive, hardware-dependent |
| Buildbot-specific | Weekly | Config drift, disk space, network |
| Phase ordering | Occasional | Pass interaction regressions |

Failure notifications are "normal" even for harmless commits — signal is diluted, real failures get missed.

## 5. Accessing Artifacts & Logs

| What | Where | How |
|------|-------|-----|
| Pre-merge CI logs | GitHub Actions → PR "Checks" tab | Click job → expand step → view log |
| Build artifacts | GitHub Actions → run → "Artifacts" section | Download zip (retained 90 days) |
| Post-commit buildbot logs | https://lab.llvm.org/buildbot/ | Find bot by name → view build log |
| Test results (lit) | Inside CI job log output | Search for `FAIL:` or `XFAIL:` |
| sccache stats | CI job log (if `--show-stats` enabled) | Search for `sccache` in log |

If a PR check fails: click the red X → find the failing job → expand the failing step. Flaky tests (lldb/openmp) often pass on re-run. Real failures show consistent `FAIL:` lines.

## 6. Automation Inventory & Gaps

| Component | Status | Gap |
|-----------|--------|-----|
| `pr-subscriber` (notify reviewers) | Active | |
| Auto-labelling (by path) | Active | |
| `clang-format` checker | Active | |
| Post-commit buildbots (200+) | Active (never green) | No flake quarantine |
| Release automation | Active | |
| Selective rebuild | Missing | Full build every PR |
| Bisection bot | Missing | Manual bisection only |
| Merge queue | Missing | 150+ commits/day vs multi-hour builds |
| Cache tracking | Missing | sccache hit rate unmonitored |

## 7. Baseline Metrics

| Metric | Current | Target |
|--------|---------|--------|
| PR first signal | ~120 min | < 30 min |
| Wall-clock (full) | 120-180 min | < 60 min |
| sccache hit rate | Unknown | > 80% |
| Flake re-runs/day | ~5+ | < 1 |
| Buildbot green rate | Never green | > 90% |

---

## Appendix A: Repository Structure

```
llvm-project/               ~9M lines total, ~2.5M C++ core
├── .github/workflows/      ~48 workflow YAML files
├── .ci/                    CI helper scripts
├── llvm/                   Core (IR, passes, codegen)
├── clang/                  C/C++/ObjC frontend
├── clang-tools-extra/      clangd, clang-tidy
├── lld/                    Linker
├── lldb/                   Debugger ← flaky tests
├── libcxx/                 C++ stdlib
├── openmp/                 OpenMP runtime ← flaky tests
├── mlir/                   Multi-Level IR
├── flang/                  Fortran frontend
├── bolt/                   Binary optimization
├── polly/                  Polyhedral optimization
└── compiler-rt/            Sanitizers, builtins
```

## Appendix B: References

| # | Resource | URL |
|---|----------|-----|
| 1 | LLVM GitHub Actions | https://github.com/llvm/llvm-project/actions |
| 2 | Workflow directory | https://github.com/llvm/llvm-project/tree/main/.github/workflows |
| 3 | CI scripts | https://github.com/llvm/llvm-project/tree/main/.ci |
| 4 | Buildbot infra | https://github.com/llvm/llvm-zorg |
| 5 | Reusable Actions | https://github.com/llvm/actions |
| 6 | Legacy pre-merge (archived) | https://github.com/google/llvm-premerge-checks |
| 7 | "LLVM: The bad parts" | https://www.npopov.com/2026/01/11/LLVM-The-bad-parts.html |
| 8 | LLVM Buildbots | https://jplehr.de/2025/01/20/llvm-buildbots/ |
| 9 | sccache idle timeout | https://www.mail-archive.com/llvm-branch-commits@lists.llvm.org/msg55844.html |
