## Libc++ LNT runners

This directory defines some LNT runners for tracking libc++ performance. A runner can be run with

```
bash <(curl -Ls https://raw.githubusercontent.com/llvm/llvm-project/main/libcxx/utils/ci/lnt/runners/RUNNER) <path-to-llvm-monorepo> [-- commit ...]
```

By default, runners poll `lnt.llvm.org` to discover un-benchmarked commits. If commits are provided
after `--`, only those commits are benchmarked and the runner exits.
