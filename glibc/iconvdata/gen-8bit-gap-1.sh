#!/bin/sh
echo "static const uint32_t iso88597_to_ucs4[96] = {"
sed -ne '/^[^[:space:]]*[[:space:]]*.x00/d;/^END/q' \
    -e 's/^<U\(....\)>[[:space:]]*.x\([A-Fa-f].\).*/  [0x\2 - 0xA0] = 0x\1,/p' \
    "$@" | sort -u
echo "};"
echo "static const struct gap from_idx[] = {"
sed -ne 's/^<U\(....\)>[[:space:]]*.x[A-Fa-f]..*/\1/p;/^END/q' \
    "$@" | sort -u | $AWK -f gap.awk
echo "  { .start = 0xffff, .end = 0xffff, .idx =     0 }"
echo "};"
echo "static const char iso88597_from_ucs4[] = {"
sed -ne 's/^<U\(....\)>[[:space:]]*.x\([A-Fa-f].\).*/0x\1 0x\2/p;/^END/q' \
    "$@" | sort -u | $AWK -f gaptab.awk
echo "};"
