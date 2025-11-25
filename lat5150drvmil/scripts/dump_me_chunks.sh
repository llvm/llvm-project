#!/usr/bin/env bash
set -euo pipefail

# Chunked ME dumper using DSMIL ioctl helper.
# Usage: sudo scripts/dump_me_chunks.sh <bytes> <output> [chunk_bytes]
# If <bytes> is "auto", uses /sys/class/.../me_region size.

SIZE_ARG="${1:-auto}"
OUT="${2:-04-hardware/microcode/me_dump.bin}"
CHUNK_ARG="${3:-65536}"
CHUNK="$CHUNK_ARG"
TMP="/tmp/me_part.bin"

read_me_region() {
  local path1="/sys/class/dsmil-84dev/dsmil-84dev/me_region"
  local path2="/sys/class/dsmil-72dev/dsmil-72dev/me_region"
  local path="$path1"
  [[ -f "$path" ]] || path="$path2"
  if [[ -f "$path" ]]; then
    read -r valid off size < <(cat "$path")
    echo "$valid" "$off" "$size"
  else
    echo "0" "0x0" "0"
  fi
}

read -r VALID OFF_HEX SIZE_SYS < <(read_me_region)
if [[ "$VALID" != "1" ]]; then
  echo "ME region is not valid. Please set me_region_override first." >&2
  exit 1
fi

if [[ "$SIZE_ARG" == "auto" ]]; then
  SIZE="$SIZE_SYS"
else
  SIZE="$SIZE_ARG"
fi

echo "Dumping ME: offset=$OFF_HEX size=${SIZE} -> $OUT (chunk=$CHUNK)"
if [[ ! -f "$OUT" ]]; then
  install -D -m 0644 /dev/null "$OUT" 2>/dev/null || true
fi
OFF=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
if (( OFF > 0 )); then echo "Resuming at +$OFF"; fi
while [[ "$OFF" -lt "$SIZE" ]]; do
  PART=$CHUNK
  if (( OFF + PART > SIZE )); then PART=$(( SIZE - OFF )); fi
  echo "Chunk +$OFF ($PART bytes)"
  python3 scripts/dsmil_ioctl_tool.py me-dump --offset "$OFF" --len "$PART" --out "$TMP"
  cat "$TMP" >> "$OUT" && rm -f "$TMP"
  OFF=$(( OFF + PART ))
done
ls -lh "$OUT"
