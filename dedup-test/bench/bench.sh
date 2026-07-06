#!/usr/bin/env bash
# Compare baseline (pre-patch) and patched clang on a modules-enabled build,
# reporting peak RSS, CPU time, and loaded SourceLocation usage.
#
# The current checkout is the patched tree; the baseline comes from a detached
# worktree at $BASELINE_REF, so the branch and index are left untouched. Both
# clangs are built Release/no-asserts with the same flags.
#
# macOS (/usr/bin/time -l via timewrap.sh).
set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
BASELINE_REF="${BASELINE_REF:-main}"
BASELINE_TREE="${BASELINE_TREE:-$REPO/../llvm-bench-baseline}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
RUNS="${RUNS:-5}"
WORKLOAD_TARGET="${WORKLOAD_TARGET:-clangBasic}"
OUT="${OUT:-$REPO/dedup-test/bench/out}"
WRAP="$REPO/dedup-test/bench/timewrap.sh"
chmod +x "$WRAP"
mkdir -p "$OUT"

echo "baseline=$BASELINE_REF  jobs=$JOBS  runs=$RUNS  workload=$WORKLOAD_TARGET"

# Baseline tree.
if [ ! -d "$BASELINE_TREE" ]; then
  git -C "$REPO" worktree add --detach "$BASELINE_TREE" "$BASELINE_REF"
fi

# Build a Release clang from a source tree.
build_clang() {  # <src> <tag>
  local src="$1" tag="$2" bdir="$1/build-bench"
  echo "building $tag clang"
  cmake -G Ninja -S "$src/llvm" -B "$bdir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=clang \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_TARGETS_TO_BUILD=Native >/dev/null
  # Building 'clang' also produces the clang++/clang-cl symlinks.
  ninja -C "$bdir" clang
}
build_clang "$REPO"          patched
build_clang "$BASELINE_TREE" baseline

# Compile the workload with one clang, timing every invocation. The first build
# populates the module cache; only subsequent clean rebuilds are recorded, so
# the numbers reflect module loading rather than first-time module building.
run_workload() {  # <clang-tree> <tag>
  local ctree="$1" tag="$2"
  local cxx="$ctree/build-bench/bin/clang++" cc="$ctree/build-bench/bin/clang"
  local wdir="$OUT/work-$tag" logdir="$OUT/log-$tag"
  rm -rf "$wdir" "$logdir"; mkdir -p "$logdir"; : > "$logdir/all.txt"

  cmake -G Ninja -S "$REPO/llvm" -B "$wdir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=clang \
    -DLLVM_ENABLE_MODULES=ON \
    -DLLVM_TARGETS_TO_BUILD=Native \
    -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
    -DCMAKE_C_COMPILER_LAUNCHER="$WRAP" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="$WRAP" >/dev/null

  # Warm the module cache. Output streams to the terminal (and to warm.log);
  # pipefail makes a compile failure here abort the run.
  echo "[$tag] warming module cache"
  ninja -C "$wdir" "$WORKLOAD_TARGET" 2>&1 | tee "$logdir/warm.log"

  for r in $(seq 1 "$RUNS"); do
    echo "[$tag] measured run $r/$RUNS"
    ninja -C "$wdir" -t clean >/dev/null 2>&1
    rm -f "$logdir"/*.m
    BENCH_LOG_DIR="$logdir" \
      ninja -C "$wdir" -j "$JOBS" "$WORKLOAD_TARGET" 2>&1 | tee "$logdir/run$r.log"
    cat "$logdir"/*.m >> "$logdir/all.txt"
  done
}
run_workload "$REPO"          patched
run_workload "$BASELINE_TREE" baseline
PATCHED_LOG="$OUT/log-patched/all.txt"
BASELINE_LOG="$OUT/log-baseline/all.txt"

# Loaded SourceLocation usage on one representative TU, for each clang.
sloc_stats() {  # <tag>
  local wdir="$OUT/work-$1" cmd
  cmd=$(ninja -C "$wdir" -t commands "$WORKLOAD_TARGET" 2>/dev/null \
        | grep -m1 -E 'clang\+\+.*\.cpp\.o') || return 0
  eval "$cmd -Xclang -print-stats" 2>&1 \
    | awk -v t="$1" '/loaded SLocEntries|de-duplicated/{print "  ["t"] "$0}'
}

python3 - "$BASELINE_LOG" "$PATCHED_LOG" <<'PY'
import sys, math, statistics as st
def load(p):
    rss=[]; cpu=[]
    for ln in open(p):
        a=ln.split()
        if len(a)==3:
            rss.append(float(a[0])/1048576.0)      # bytes -> MB
            cpu.append(float(a[1])+float(a[2]))     # user+sys
    return rss,cpu
gm=lambda xs: math.exp(sum(map(math.log,xs))/len(xs)) if xs else 0.0
br,bc=load(sys.argv[1]); pr,pc=load(sys.argv[2])
def row(n,b,p,f):
    d=(p-b)/b*100 if b else 0.0
    print(f"{n:<32}{f(b):>12}{f(p):>12}{d:>+9.1f}%")
mb=lambda x:f"{x:,.0f}"; s=lambda x:f"{x:.3f}"
print(f"\n{'':<32}{'baseline':>12}{'patched':>12}{'delta':>10}")
print("-"*66)
print(f"compiles: {len(bc)} baseline / {len(pc)} patched")
row("peak RSS, max cc1 (MB)", max(br), max(pr), mb)
row("peak RSS, geomean/TU (MB)", gm(br), gm(pr), mb)
row("CPU time, total (s)", sum(bc), sum(pc), s)
row("CPU time, geomean/TU (s)", gm(bc), gm(pc), s)
row("CPU time, median/TU (s)", st.median(bc), st.median(pc), s)
PY

echo
echo "loaded SourceLocation usage (one TU):"
sloc_stats baseline || true
sloc_stats patched  || true

echo
echo "logs: $OUT/log-{baseline,patched}/all.txt"
echo "drop the baseline tree with: git worktree remove $BASELINE_TREE"
