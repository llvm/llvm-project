#!/usr/bin/env bash
# Compiler launcher. Runs the compile under `time -l` and, when BENCH_LOG_DIR
# is set, records peak RSS (bytes) and user+sys time (seconds) for it there.
# With BENCH_LOG_DIR unset (e.g. the warm build) it just runs the compile.
set -u

t=$(mktemp)
/usr/bin/time -l "$@" 2>"$t"
rc=$?

rss=$(awk '/maximum resident set size/{print $1}' "$t")
read -r usr sys < <(awk '/real/&&/user/&&/sys/{print $3, $5}' "$t")

# Record only measured builds, and only for real compiles (link/archive steps
# have no RSS line).
if [ -n "${BENCH_LOG_DIR:-}" ] && [ -n "${rss:-}" ]; then
  printf '%s %s %s\n' "$rss" "${usr:-0}" "${sys:-0}" \
    > "$BENCH_LOG_DIR/$(date +%s)-$$-$RANDOM.m"
fi

# Forward the compiler's own stderr; drop the trailing resource block.
awk '/real/&&/user/&&/sys/{stop=1} !stop{print}' "$t" >&2
rm -f "$t"
exit $rc
