#!/bin/sh
echo "static const uint8_t to_ucs1[256] = {"
sed -ne '/^[^[:space:]]*[[:space:]]*.x00/d;/^END/q' \
    -e 's/^<U00\(..\)>[[:space:]]*.x\(..\).*/  [0x\2] = 0x\1,/p' \
    "$@" | sort -u
echo "};"
