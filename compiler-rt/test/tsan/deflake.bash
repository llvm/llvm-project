#!/usr/bin/env bash
# This script is used to deflake inherently flaky tsan tests.
# It is invoked from lit tests as:
# %deflake $THRESHOLD  mybinary
# which is then substituted by lit to:
# $(dirname %s)/deflake.bash $THRESHOLD mybinary
# - When TSAN_TEST_DEFLAKE_THRESHOLD is defined to a positive integer value,
#   THRESHOLD will be the defined value.
# - When TSAN_TEST_DEFLAKE_THRESHOLD is not defined, THRESHOLD will be 10.
# The script runs the target program up to $THRESHOLD times,
# until it fails (i.e. produces a race report).

THRESHOLD="${1}"
shift

# Early exit if $THRESHOLD is not a non-negative integer
[[  "${THRESHOLD}" =~ ^[0-9]+$ ]] || exit 1

while (( THRESHOLD-- )); do
	OUT=`$@ 2>&1`
	if [[ $? != 0 ]]; then
		echo "$OUT"
		exit 0
	fi
done
exit 1
