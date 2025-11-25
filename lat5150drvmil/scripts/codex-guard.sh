
#!/usr/bin/env bash
# codex-guard.sh — GPT‑5 Codex CLI wrapper (full output, auditing, safe tooling)
set -euo pipefail

CFG="${CFG:-codexrc.json}"
[ -f "$CFG" ] || { echo "[ERR] missing $CFG"; exit 1; }
command -v jq >/dev/null || { echo "[ERR] jq required"; exit 1; }

CODEX_BIN="${CODEX_BIN:-codex}"

ENGINE=$(jq -r '.engine' "$CFG")
LOG_DIR=$(jq -r '.log_dir' "$CFG")
TRANSCRIPT=$(jq -r '.transcript_file' "$CFG")
EVENTS_CSV=$(jq -r '.events_csv' "$CFG")
EVENTS_JSONL=$(jq -r '.events_jsonl' "$CFG")
MAX_STEP=$(jq -r '.max_tokens_per_step' "$CFG")
SOFT_OUT_CAP=$(jq -r '.soft_output_tokens_cap' "$CFG")
STEP_CAP=$(jq -r '.session_step_cap' "$CFG")
PREFACE=$(jq -r '.preface_prompt' "$CFG")
PROJECT=$(jq -r '.project // "PROJECT"' "$CFG")
MODE=$(jq -r '.mode' "$CFG")
# Extra CLI args for codex binary (space-separated string). Example:
# "--dangerously-bypass-approvals-and-sandbox --resume"
EXTRA_ARGS_STR=$(jq -r '.extra_cli_args // ""' "$CFG")
read -r -a EXTRA_ARGS <<< "$EXTRA_ARGS_STR"

# Force full, non-truncated output in this session
export PAGER=cat
export LESS='-R -S -F -X'

mkdir -p "$LOG_DIR"
: > "$TRANSCRIPT"
: > "$EVENTS_JSONL"
printf '"ts","project","event","engine","input_tokens","output_tokens","note"\n' > "$EVENTS_CSV"

mapfile -t REDACT < <(jq -r '.redact_patterns[]' "$CFG")
mapfile -t ALLOW  < <(jq -r '.allowed_commands[]' "$CFG")
mapfile -t DENYRX < <(jq -r '.denied_patterns[]' "$CFG")

redact_stream() {
  local tmp; tmp="$(mktemp)"; cat > "$tmp"
  for rx in "${REDACT[@]}"; do perl -0777 -pe "s/$rx/[REDACTED]/g" -i "$tmp" || true; done
  cat "$tmp"; rm -f "$tmp"
}

log_csv() {
  local ts; ts="$(date -Iseconds)"
  printf '"%s","%s","%s","%s","%s","%s","%s"\n' "$ts" "$PROJECT" "$1" "$ENGINE" "${2:-0}" "${3:-0}" "${4:-}" >> "$EVENTS_CSV"
}

append_transcript() {
  {
    echo "----- $(date -Iseconds) [$PROJECT] -----"
    echo "[ENGINE] $ENGINE | [MODE] $MODE"
    echo "[PREFACE] $PREFACE"
    echo "[PROMPT]"
    cat "$1"
    echo "----------------------------------------"
  } >> "$TRANSCRIPT"
}

apply_patches() {
  local json="$1"
  jq -c '.[]' <<<"$json" | while read -r patch; do
    local f sl el rep
    f=$(jq -r '.file' <<<"$patch")
    sl=$(jq -r '.start_line // empty' <<<"$patch")
    el=$(jq -r '.end_line   // empty' <<<"$patch")
    rep=$(jq -r '.replacement // ""' <<<"$patch")

    [ -n "$f" ] || { echo "[PATCH] skip: missing file"; continue; }
    [ -f "$f" ] || { echo "[PATCH] create: $f"; touch "$f"; }

    if [ -n "$sl" ] && [ -n "$el" ]; then
      tmp="$(mktemp)"
      nl -ba "$f" | awk -v s="$sl" -v e="$el" 'NR<s || NR>e {print substr($0, index($0,$2))}' > "$tmp"
      awk -v s="$sl" 'NR==s-1{print; print ENVIRON["AIOPS_REP"]; next}1' AIOPS_REP="$rep" "$tmp" > "$f.new"
      mv "$f.new" "$f"
      rm -f "$tmp"
      echo "[PATCH] $f:$sl-$el replaced"
    else
      local find; find=$(jq -r '.find // empty' <<<"$patch")
      if [ -n "$find" ]; then
        perl -0777 -pe "s/\Q$find\E/$rep/g" -i "$f"
        echo "[PATCH] $f regex replace"
      else
        echo "[PATCH] skip: no range or find"
      fi
    fi
  done
}

run_codex() {
  local prompt_file="$1"
  append_transcript "$prompt_file"

  local OUT_JSON; OUT_JSON="$("$CODEX_BIN" "${EXTRA_ARGS[@]}" --model "$ENGINE" --max-tokens "$MAX_STEP" --preface "$PREFACE" --json < "$prompt_file")" || {
    echo "[ERR] codex failed"; exit 1;
  }
  echo "$OUT_JSON" >> "$EVENTS_JSONL"

  local in_t out_t; in_t=$(jq -r '.usage.input_tokens  // 0' <<<"$OUT_JSON")
  out_t=$(jq -r '.usage.output_tokens // 0' <<<"$OUT_JSON")
  log_csv "step" "$in_t" "$out_t" "ok"

  if jq -e '.refused == true' <<<"$OUT_JSON" >/dev/null 2>&1; then
    if jq -e '.refusal_retry.enabled' "$CFG" >/dev/null; then
      local maxr; maxr=$(jq -r '.refusal_retry.max_retries' "$CFG")
      local refr; refr=$(jq -r '.refusal_retry.defensive_reframe' "$CFG")
      if [ "${maxr:-0}" -gt 0 ]; then
        echo "[INFO] refusal → applying defensive reframe & retry once"
        {
          echo "$refr"
          echo
          cat "$prompt_file"
        } > "$prompt_file.reframed"
        OUT_JSON="$("$CODEX_BIN" "${EXTRA_ARGS[@]}" --model "$ENGINE" --max-tokens "$MAX_STEP" --preface "$PREFACE" --json < "$prompt_file.reframed")"
        echo "$OUT_JSON" >> "$EVENTS_JSONL"
        in_t=$(jq -r '.usage.input_tokens  // 0' <<<"$OUT_JSON")
        out_t=$(jq -r '.usage.output_tokens // 0' <<<"$OUT_JSON")
        log_csv "retry" "$in_t" "$out_t" "reframed"
      fi
    fi
  fi

  echo "[ASSISTANT]:" >> "$TRANSCRIPT"
  jq -r '.text // ""' <<<"$OUT_JSON" | tee >(redact_stream >> "$TRANSCRIPT") >/dev/tty

  if [ "${out_t:-0}" -gt "$SOFT_OUT_CAP" ]; then
    echo "[NOTE] Large output; consider chunking or retrieval."
  fi

  if jq -e '.patches' <<<"$OUT_JSON" >/dev/null 2>&1; then
    local PJSON; PJSON="$(jq -c '.patches' <<<"$OUT_JSON")"
    apply_patches "$PJSON"
  fi

  if jq -e '.commands' <<<"$OUT_JSON" >/dev/null 2>&1; then
    mapfile -t CMDS < <(jq -r '.commands[]' <<<"$OUT_JSON")
    for c in "${CMDS[@]}"; do
      for rx in "${DENYRX[@]}"; do
        if [[ "$c" =~ $rx ]]; then echo "[BLOCK] $c"; exit 3; fi
      done
      local ok=0
      for a in "${ALLOW[@]}"; do
        if [[ "$c" == "$a"* ]]; then ok=1; break; fi
      done
      (( ok==1 )) || { echo "[DENY] $c (not in allowlist)"; exit 3; }
      echo "[RUN] $c" | tee -a "$TRANSCRIPT"
      bash -lc "$c" 2>&1 | tee -a "$TRANSCRIPT"
    done
  fi
}

echo "[READY] Codex guard (GPT-5 CLI, full output). Ctrl+C to exit."
STEP=0
PROMPT_FILE="$(mktemp)"; trap 'rm -f "$PROMPT_FILE"' EXIT

while true; do
  (( STEP++ ))
  if (( STEP > STEP_CAP )); then echo "[HALT] Step cap reached ($STEP_CAP)."; log_csv "halt" 0 0 "step_cap"; exit 0; fi
  printf "\n[user %03d] > " "$STEP"
  if ! IFS= read -r line; then echo; exit 0; fi
  echo "$line" > "$PROMPT_FILE"
  run_codex "$PROMPT_FILE"
done
