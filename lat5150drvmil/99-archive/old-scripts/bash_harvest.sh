#!/usr/bin/env bash
# mcu_harvest.sh — Acquire Intel microcode blob for Meteor Lake (or given CPUID) at a target or max revision.
# - Prefers rev 0x1c, else selects highest < --max-rev (default 0x22).
# - Outputs: ./mcu_out/<name>.bin + manifest.json
# Requirements: git, coreutils (od, sha256sum, stat), jq (for manifest). Optional: iucode-tool.
set -euo pipefail

# Defaults
CPUID="06-aa-04"
PREFERRED_REV_HEX="0x1c"
MAX_REV_HEX="0x22"      # accept anything strictly less than this
OUTDIR="./mcu_out"
REPO_URL="https://github.com/intel/Intel-Linux-Processor-Microcode-Data-Files.git"
WORKDIR="$(mktemp -d -t mcu-harvest-XXXXXX)"
CLEANUP=1

# --- Arg parse (simple) ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpuid) CPUID="${2:?}"; shift 2;;
    --max-rev) MAX_REV_HEX="${2:?}"; shift 2;;
    --preferred-rev) PREFERRED_REV_HEX="${2:?}"; shift 2;;
    --out|--outdir) OUTDIR="${2:?}"; shift 2;;
    --repo) REPO_URL="${2:?}"; shift 2;;
    --noclean) CLEANUP=0; shift;;
    -h|--help)
      cat <<EOF
Usage: $0 [--cpuid 06-aa-04] [--max-rev 0x22] [--preferred-rev 0x1c] [--out ./mcu_out] [--repo URL] [--noclean]
Acquires Intel microcode blob from official repo by scanning release tags and picking the preferred or highest < max-rev.
EOF
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$OUTDIR"

# --- Helpers ---
hex_to_dec() { # $1 like 0x1c
  printf "%d" "$(( $1 ))"
}
dec_to_hex8() { # to 0x0000001c
  printf "0x%08x" "$1"
}

# Read Intel microcode header fields (DWORDs, little endian):
# DWORD 0: header_version (0x00000001)
# DWORD 1: update_revision (this is the microcode rev)
# DWORD 2: date (yyyymmdd BCD-like int, from Intel format)
# DWORD 3: processor_signature (CPUID)
# DWORD 6: platform_id (bitmask)
read_header_fields() {
  local f="$1"
  # Grab first 28 bytes (7 DWORDs); use od to extract words
  # Second dword = rev, fourth = sig, seventh = platform
  local words; words=$(od -An -t x4 -N 28 "$f" | xargs)
  # words = 7 hex words: w0 w1 w2 w3 w4 w5 w6
  read -r w0 w1 w2 w3 w4 w5 w6 <<<"$words"
  # Normalize lowercase
  w1=${w1,,}; w3=${w3,,}; w6=${w6,,}
  echo "$w1 $w3 $w6"
}

pick_file_rev_hex() {
  # returns e.g. 0x0000001c
  local f="$1"; local rev sig pf
  read -r rev sig pf < <(read_header_fields "$f")
  printf "0x%s\n" "${rev#0x}" 2>/dev/null || echo "0x$rev"
}

pick_file_sig_hex() {
  local f="$1"; local rev sig pf
  read -r rev sig pf < <(read_header_fields "$f")
  printf "0x%s\n" "${sig#0x}"
}

pick_file_pf_hex() {
  local f="$1"; local rev sig pf
  read -r rev sig pf < <(read_header_fields "$f")
  printf "0x%s\n" "${pf#0x}"
}

# --- Clone repo (shallow tags for speed) ---
echo "[*] Cloning Intel microcode repo (tags only) → $WORKDIR"
git clone --quiet --filter=tree:0 --no-checkout "$REPO_URL" "$WORKDIR"
cd "$WORKDIR"
git fetch --quiet --tags

# Get tag list (release-style), sorted
mapfile -t TAGS < <(git tag -l 'microcode-*' | sort)

if [[ ${#TAGS[@]} -eq 0 ]]; then
  echo "[!] No release tags found. Aborting." >&2
  exit 3
fi

# We will search for the file at intel-ucode/<CPUID> in each tag,
# collect candidates with rev < MAX_REV_HEX, prefer exact PREFERRED_REV_HEX, else max < MAX_REV
pref_dec=$(hex_to_dec "$PREFERRED_REV_HEX")
max_dec=$(hex_to_dec "$MAX_REV_HEX")

best_tag=""
best_rev_dec=-1
best_file_tmp=""

echo "[*] Scanning tags for CPUID ${CPUID} with preferred ${PREFERRED_REV_HEX} and max < ${MAX_REV_HEX}..."

for tag in "${TAGS[@]}"; do
  # Sparse checkout the specific path for speed
  git checkout -q "$tag" || continue
  if [[ -f "intel-ucode/${CPUID}" ]]; then
    f="intel-ucode/${CPUID}"
    rev_hex=$(pick_file_rev_hex "$f")
    rev_dec=$(hex_to_dec "$rev_hex")
    # Filter: rev < MAX
    if (( rev_dec < max_dec )); then
      printf "[+] %s: found %s rev=%s\n" "$tag" "$f" "$(dec_to_hex8 "$rev_dec")"
      # Check for exact preferred
      if (( rev_dec == pref_dec )); then
        best_tag="$tag"
        best_rev_dec="$rev_dec"
        best_file_tmp="$f"
        echo "[*] Exact preferred revision found at tag $tag → $(dec_to_hex8 "$best_rev_dec")"
        break
      fi
      # Otherwise keep the highest < max
      if (( rev_dec > best_rev_dec )); then
        best_tag="$tag"
        best_rev_dec="$rev_dec"
        best_file_tmp="$f"
      fi
    else
      printf "[-] %s: %s rev=%s is >= max; skipping\n" "$tag" "$f" "$(dec_to_hex8 "$rev_dec")"
    fi
  fi
done

if [[ -z "${best_tag}" ]]; then
  echo "[!] No matching microcode blob < ${MAX_REV_HEX} found for CPUID ${CPUID}." >&2
  (( CLEANUP )) && rm -rf "$WORKDIR" || true
  exit 4
fi

# Prepare output
commit_id=$(git rev-parse --short "$best_tag")
outfile="${OUTDIR}/MTL_${CPUID}_rev$(dec_to_hex8 "$best_rev_dec")_${best_tag}_${commit_id}.bin"
manifest="${OUTDIR}/manifest.json"

cp -f "$best_file_tmp" "$outfile"

# Collect metadata
sig_hex=$(pick_file_sig_hex "$outfile")
pf_hex=$(pick_file_pf_hex("$outfile") 2>/dev/null || true) # shellcheck disable=SC2091
if [[ -z "${pf_hex:-}" ]]; then
  # fallback if subshell form above unhappy on some shells
  read -r _ sig_hex pf_hex < <(read_header_fields "$outfile")
  sig_hex="0x${sig_hex,,}"
  pf_hex="0x${pf_hex,,}"
fi
size_bytes=$(stat -c%s "$outfile")
sha256=$(sha256sum "$outfile" | awk '{print $1}')
now_iso=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

echo "[*] Selected:"
echo "    Tag:        $best_tag ($commit_id)"
echo "    File:       $outfile"
echo "    Revision:   $(dec_to_hex8 "$best_rev_dec")"
echo "    Signature:  ${sig_hex}"
echo "    PlatformID: ${pf_hex}"
echo "    Size:       ${size_bytes} bytes"
echo "    SHA256:     ${sha256}"

# Write manifest
jq -n --arg cpuid "$CPUID" \
      --arg tag "$best_tag" \
      --arg commit "$commit_id" \
      --arg rev "$(dec_to_hex8 "$best_rev_dec")" \
      --arg sig "$sig_hex" \
      --arg pf  "$pf_hex" \
      --arg size "$size_bytes" \
      --arg sha "$sha256" \
      --arg path "$outfile" \
      --arg repo "$REPO_URL" \
      --arg generated "$now_iso" \
'{
  cpuid: $cpuid,
  chosen_tag: $tag,
  commit: $commit,
  revision: $rev,
  signature: $sig,
  platform_mask: $pf,
  size_bytes: ($size|tonumber),
  sha256: $sha,
  output_path: $path,
  source_repo: $repo,
  generated_utc: $generated
}' > "$manifest"

echo "[*] Manifest written: $manifest"

# Optional: verify with iucode-tool if present
if command -v iucode_tool >/dev/null 2>&1; then
  echo "[*] iucode-tool check:"
  iucode_tool -tb "$outfile" || true
else
  echo "[i] iucode-tool not installed; skipping additional decode."
fi

# Cleanup
if (( CLEANUP )); then
  rm -rf "$WORKDIR"
fi

echo "[✓] Done."
