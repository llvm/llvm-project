#!/bin/sh
echo "static const uint32_t to_ucs4[256] = {"
sed -ne '/^[^[:space:]]*[[:space:]]*.x00/d;/^END/q' \
    -e 's/^<U\(....\)>[[:space:]]*.x\(..\).*/  [0x\2] = 0x\1,/p' \
    "$@" | sort -u
echo "};"
echo "static const struct gap from_idx[] = {"
sed -ne 's/^<U\(....\).*/\1/p;/^END/q' \
    "$@" | sort -u | $AWK -f gap.awk
echo "  { .start = 0xffff, .end = 0xffff, .idx =     0 }"
echo "};"
echo "static const char from_ucs4[] = {"
sed -ne 's/^<U\(....\)>[[:space:]]*.x\(..\).*/\1 \2/p;/^END/q' \
    "$@" | sort -u | $AWK -f gaptab.awk
echo "};"
