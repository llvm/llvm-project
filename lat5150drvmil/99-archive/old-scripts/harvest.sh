5#!/usr/bin/env bash
# mcu_harvest.sh — Stream-only Intel microcode harvester (no git checkout)
# - Finds release tags via `git ls-remote`.
# - Streams each tag tarball and extracts ONLY intel-ucode/<CPUID> via tar -xO (stdout).
# - Prefers --exact or --preferred-rev (default 0x1c); else picks highest < --max-rev (default 0x22).
# - Outputs: <OUTDIR>/MTL_<CPUID>_rev<hex>_<tag>_<short>.bin + manifest.json
# Requirements: bash, git, curl, tar, coreutils (od, sha256sum, stat), awk, sed; jq (optional), iucode-tool (optional).
set -euo pipefail

CPUID="06-aa-04"
PREFERRED_REV_HEX="0x1c"
MAX_REV_HEX="0x22"    # strictly less than
EXACT_REV=""
OUTDIR="$PWD/mcu_out"
REPO_WEB="https://github.com/intel/Intel-Linux-Processor-Microcode-Data-Files"
REPO_GIT="https://github.com/intel/Intel-Linux-Processor-Microcode-Data-Files.git"
QUIET=0

usage(){ cat <<EOF
Usage: $0 [--cpuid 06-aa-04] [--preferred-rev 0x1c] [--max-rev 0x22] [--exact 0x1c] [--out DIR] [-q]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpuid) CPUID="${2:?}"; shift 2;;
    --preferred-rev) PREFERRED_REV_HEX="${2:?}"; shift 2;;
    --max-rev) MAX_REV_HEX="${2:?}"; shift 2;;
    --exact) EXACT_REV="${2:?}"; shift 2;;
    --out|--outdir) OUTDIR="${2:?}"; shift 2;;
    -q|--quiet) QUIET=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

log(){ (( QUIET )) || echo -e "$*"; }

need(){ command -v "$1" >/dev/null 2>&1 || { echo "[!] Missing tool: $1" >&2; exit 10; }; }
need git; need curl; need tar; need od; need sha256sum; need awk; need sed; need stat

mkdir -p "$OUTDIR" || { echo "[!] Cannot create OUTDIR: $OUTDIR" >&2; exit 11; }
OUTDIR="$(cd "$OUTDIR" && pwd -P)"
log "[*] Output dir: $OUTDIR"
log "[*] Target CPUID: $CPUID  | preferred: ${PREFERRED_REV_HEX}${EXACT_REV:+ (exact)} | max: < ${MAX_REV_HEX}"

hex_to_dec(){ printf "%d" "$(( $1 ))"; }
dec_to_hex8(){ printf "0x%08x" "$1"; }
get_rev_hex(){ od -An -t x4 -N 8 "$1" | awk '{print "0x"$2}' | tr 'A-F' 'a-f'; }     # dword1
get_sig_hex(){ od -An -t x4 -N 16 "$1" | awk '{print "0x"$4}' | tr 'A-F' 'a-f'; }    # dword3
get_pf_hex(){  od -An -t x4 -N 28 "$1" | awk '{print "0x"$7}' | tr 'A-F' 'a-f'; }    # dword6

# List release tags (no objects fetched)
log "[*] Enumerating release tags (no checkout)…"
mapfile -t TAGS < <(git ls-remote --tags --refs "$REPO_GIT" 'microcode-*' | awk '{print $2}' | sed 's#refs/tags/##' | sort)
((${#TAGS[@]})) || { echo "[!] No release tags found." >&2; exit 12; }

pref_dec=$(hex_to_dec "${EXACT_REV:-$PREFERRED_REV_HEX}")
max_dec=$(hex_to_dec "$MAX_REV_HEX")

best_tag=""; best_rev_dec=-1; best_tmp=""
tmpdir="$(mktemp -d -t mcu-stream-XXXXXX)"
trap 'rm -rf "$tmpdir"' EXIT

# Helper: try a single tag by streaming tarball and extracting CPUID file to temp
try_tag(){
  local tag="$1" tmpf="$2" url
  # Use codeload (reliable, CDN-backed)
  url="https://codeload.github.com/intel/Intel-Linux-Processor-Microcode-Data-Files/tar.gz/refs/tags/${tag}"
  # Extract just the file; wildcard because tarball topdir name varies
  if ! curl -fsSL --retry 4 --retry-delay 1 "$url" \
      | tar -xOzf - --wildcards "*/intel-ucode/${CPUID}" > "$tmpf" 2>/dev/null; then
    return 1
  fi
  [[ -s "$tmpf" ]] || return 1
  return 0
}

log "[*] Scanning tags for intel-ucode/${CPUID}…"
for tag in "${TAGS[@]}"; do
  tmp="$tmpdir/${tag//\//_}_${CPUID}.bin"
  if try_tag "$tag" "$tmp"; then
    rev_hex=$(get_rev_hex "$tmp")
    rev_dec=$(hex_to_dec "$rev_hex")
    if (( rev_dec < max_dec )); then
      log "[+] $tag: rev $(dec_to_hex8 "$rev_dec")"
      if [[ -n "$EXACT_REV" ]] && (( rev_dec == pref_dec )); then
        best_tag="$tag"; best_rev_dec="$rev_dec"; best_tmp="$tmp"; log "[*] Exact revision matched."; break
      fi
      if [[ -z "$EXACT_REV" ]] && (( rev_dec == pref_dec )); then
        best_tag="$tag"; best_rev_dec="$rev_dec"; best_tmp="$tmp"; log "[*] Exact preferred revision found."; break
      fi
      if [[ -z "$EXACT_REV" ]] && (( rev_dec > best_rev_dec )); then
        best_tag="$tag"; best_rev_dec="$rev_dec"; best_tmp="$tmp"
      fi
    else
      log "[-] $tag: rev $(dec_to_hex8 "$rev_dec") >= $(dec_to_hex8 "$max_dec"); skip"
    fi
  fi
done

if [[ -z "$best_tag" ]]; then
  if [[ -n "$EXACT_REV" ]]; then
    echo "[!] Exact revision $EXACT_REV not found (< $MAX_REV_HEX) for CPUID $CPUID." >&2
  else
    echo "[!] No qualifying microcode (< $MAX_REV_HEX) found for CPUID $CPUID." >&2
  fi
  exit 13
fi

# Write final artifact + manifest
short="$(git ls-remote --tags --refs "$REPO_GIT" "$best_tag" | awk '{print substr($1,1,7)}')"
outfile="${OUTDIR}/MTL_${CPUID}_rev$(dec_to_hex8 "$best_rev_dec")_${best_tag}_${short}.bin"
cp -f "$best_tmp" "$outfile"

sig_hex=$(get_sig_hex "$outfile")
pf_hex=$(get_pf_hex "$outfile")
size_bytes=$(stat -c%s "$outfile")
sha256=$(sha256sum "$outfile" | awk '{print $1}')
now_iso=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
manifest="${OUTDIR}/manifest.json"

log "[*] Selected:"
log "    Tag:        $best_tag ($short)"
log "    File:       $outfile"
log "    Revision:   $(dec_to_hex8 "$best_rev_dec")"
log "    Signature:  $sig_hex"
log "    PlatformID: $pf_hex"
log "    Size:       ${size_bytes} bytes"
log "    SHA256:     $sha256"

if command -v jq >/dev/null 2>&1; then
  jq -n --arg cpuid "$CPUID" --arg tag "$best_tag" --arg commit "$short" \
        --arg rev "$(dec_to_hex8 "$best_rev_dec")" --arg sig "$sig_hex" --arg pf "$pf_hex" \
        --argjson size "$size_bytes" --arg sha "$sha256" --arg path "$outfile" \
        --arg repo "$REPO_WEB" --arg generated "$now_iso" \
        '{cpuid:$cpuid,chosen_tag:$tag,commit:$commit,revision:$rev,signature:$sig,
          platform_mask:$pf,size_bytes:$size,sha256:$sha,output_path:$path,
          source_repo:$repo,generated_utc:$generated}' > "$manifest"
else
  cat > "$manifest" <<JSON
{"cpuid":"$CPUID","chosen_tag":"$best_tag","commit":"$short","revision":"$(dec_to_hex8 "$best_rev_dec")","signature":"$sig_hex","platform_mask":"$pf_hex","size_bytes":$size_bytes,"sha256":"$sha256","output_path":"$outfile","source_repo":"$REPO_WEB","generated_utc":"$now_iso"}
JSON
fi
[[ -s "$manifest" ]] || { echo "[!] Manifest write failed" >&2; exit 14; }

if command -v iucode_tool >/dev/null 2>&1; then
  log "[*] iucode-tool decode:"; iucode_tool -tb "$outfile" || true
fi

log "[✓] Done."

