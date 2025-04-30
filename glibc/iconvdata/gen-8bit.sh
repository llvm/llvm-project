#!/bin/sh
echo "static const uint32_t to_ucs4[256] = {"
sed -ne '/^[^[:space:]]*[[:space:]]*.x00/d;/^END/q' \
    -e 's/^<U\(....\)>[[:space:]]*.x\(..\).*/  [0x\2] = 0x\1,/p' \
    "$@" | sort -u
echo "};"
echo "static const char from_ucs4[] = {"
sed -ne '/^[^[:space:]]*[[:space:]]*.x00/d;/^END/q' \
    -e 's/^<U\(....\)>[[:space:]]*.x\(..\).*/  [0x\1] = 0x\2,/p' \
    "$@" | sort -u
echo "};"
