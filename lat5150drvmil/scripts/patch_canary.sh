#!/usr/bin/env bash
set -euo pipefail

# Surgical canary patch (with immediate reversion) using DSMIL ME patch ioctls.
#
# Usage: sudo scripts/patch_canary.sh <abs_offset_hex> <length_dec>
# Example: sudo scripts/patch_canary.sh 0x81b400 64
#
# - Enables service_mode gates
# - Sets me_slack_override to target window
# - Backs up original bytes to /tmp/me_canary_backup.bin
# - Writes canary payload ("DSMIL_CANARY_V1\n" padded) via me-patch
# - Verifies readback
# - Reverts by writing the backup back via me-patch

ABS_OFF_STR="${1:?abs offset (hex)}"
LEN_STR="${2:?length (dec)}"

abs_off=$((ABS_OFF_STR))
len=$((LEN_STR))

REG_PATH="/sys/class/dsmil-84dev/dsmil-84dev/me_region"
[ -f "$REG_PATH" ] || REG_PATH="/sys/class/dsmil-72dev/dsmil-72dev/me_region"

if [[ ! -f "$REG_PATH" ]]; then
  echo "me_region sysfs not found" >&2
  exit 1
fi

read valid base size < "$REG_PATH"
if [[ "$valid" != "1" ]]; then
  echo "me_region is not valid; set me_region_override first" >&2
  exit 1
fi

base_off=$((base))
slack_off=$((abs_off - base_off))

echo "Target abs_off=0x$(printf %x $abs_off) (rel=0x$(printf %x $slack_off)) len=$len"

ME_ACCESS="/sys/class/dsmil-84dev/dsmil-84dev/service_mode_me_access"
[ -f "$ME_ACCESS" ] || ME_ACCESS="/sys/class/dsmil-72dev/dsmil-72dev/service_mode_me_access"
echo 1 | tee "$ME_ACCESS" >/dev/null

SLACK_PATH="/sys/class/dsmil-84dev/dsmil-84dev/me_slack_override"
[ -f "$SLACK_PATH" ] || SLACK_PATH="/sys/class/dsmil-72dev/dsmil-72dev/me_slack_override"
echo "0x$(printf %x $slack_off) $len" | tee "$SLACK_PATH" >/dev/null

# Backup original bytes
BACKUP="/tmp/me_canary_backup.bin"
python3 scripts/dsmil_ioctl_tool.py me-dump --absolute --offset "$abs_off" --len "$len" --out "$BACKUP"

# Build canary payload (pad to len)
CANARY="/tmp/me_canary_payload.bin"
printf 'DSMIL_CANARY_V1\n' > "$CANARY"
python3 - <<PY
from pathlib import Path
import sys
p=Path("$CANARY"); data=p.read_bytes()
data=(data*((( $len + len(data)-1)//len(data)) ))[:$len]
p.write_bytes(data)
PY

# Compute CRC32 for gate
CRC=$(python3 - <<PY
import binascii
print(hex(binascii.crc32(open("$CANARY","rb").read()) & 0xffffffff))
PY
)

PATCH_GATE="/sys/class/dsmil-84dev/dsmil-84dev/service_mode_me_patch"
[ -f "$PATCH_GATE" ] || PATCH_GATE="/sys/class/dsmil-72dev/dsmil-72dev/service_mode_me_patch"
echo "1 $CRC" | tee "$PATCH_GATE" >/dev/null

# Apply canary
python3 scripts/dsmil_ioctl_tool.py me-patch --input "$CANARY" --slack-offset 0

# Verify canary
READBACK="/tmp/me_canary_readback.bin"
python3 scripts/dsmil_ioctl_tool.py me-dump --absolute --offset "$abs_off" --len "$len" --out "$READBACK"
if ! cmp -s "$CANARY" "$READBACK"; then
  echo "Canary verify failed" >&2
  exit 2
fi
echo "Canary verify OK"

# Revert to original
CRC2=$(python3 - <<PY
import binascii
print(hex(binascii.crc32(open("$BACKUP","rb").read()) & 0xffffffff))
PY
)
echo "1 $CRC2" | tee "$PATCH_GATE" >/dev/null
python3 scripts/dsmil_ioctl_tool.py me-patch --input "$BACKUP" --slack-offset 0

# Verify revert
python3 scripts/dsmil_ioctl_tool.py me-dump --absolute --offset "$abs_off" --len "$len" --out "$READBACK"
if ! cmp -s "$BACKUP" "$READBACK"; then
  echo "Reversion verify failed" >&2
  exit 3
fi
echo "Reversion verify OK"

rm -f "$READBACK" "$CANARY"
echo "Done"

