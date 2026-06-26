#!/usr/bin/env python3
"""
ejit_strip_sched_models.py — Null out per-CPU scheduling model table pointers
in the TableGen-generated AArch64GenSubtargetInfo.inc.

Background
----------
AArch64GenSubtargetInfo.inc defines ~24 per-CPU scheduling models, each holding
a ~24 KB MCSchedClassDesc array and a smaller MCProcResourceDesc array.  All
models are kept alive by AArch64SubTypeKV (the CPU-name lookup table used during
subtarget initialization), so gc-sections cannot eliminate them through normal
linkage.

This script patches three fields in every MCSchedModel initializer that should
be stripped:
  ProcResourceTable        → nullptr
  SchedClassTable          → nullptr
  NumProcResourceKinds     → 0
  NumSchedClasses          → 0

After patching, those arrays become unreferenced and gc-sections in the lipo
pipeline eliminates them.

Usage
-----
    python3 ejit_strip_sched_models.py BUILD_DIR [KEEP_CPU]

  BUILD_DIR   Path to the cmake build directory (e.g. build_release_aarch64).
  KEEP_CPU    LLVM CPU name whose scheduling model should be preserved intact.
              All other non-generic models are stripped.  Omit to strip every
              model.

              Examples:
                "cortex-a57"   → keeps CortexA57Model (also used by a72/a73/a75–a78c)
                "neoverse-n2"  → keeps NeoverseN2Model
                ""  or omit   → strips all models

After patching, rebuild the two libraries that include the .inc file:
    ninja -C BUILD_DIR LLVMAArch64CodeGen LLVMAArch64Desc

To restore the original .inc, delete it and run:
    ninja -C BUILD_DIR AArch64CommonTableGen
run_aarch64_pipeline.sh does this automatically when switching to no-strip mode.
"""

import sys
import re
import os

INC_PATH = "lib/Target/AArch64/AArch64GenSubtargetInfo.inc"
SENTINEL = "// EJIT: stripped"


def find_keep_model(content: str, keep_cpu: str) -> str | None:
    """Return the MCSchedModel variable name mapped to keep_cpu in AArch64SubTypeKV."""
    kv_start = content.find('AArch64SubTypeKV[]')
    if kv_start == -1:
        return None
    kv_end = content.find('\n};', kv_start)
    if kv_end == -1:
        return None
    kv_block = content[kv_start:kv_end]

    # Each KV entry is one line: { "cpu-name", {...}, {...}, &ModelName }
    pos = kv_block.find(f'"{keep_cpu}"')
    if pos == -1:
        return None

    # Scan forward from the CPU string to the end of its entry.
    m = re.search(r'&(\w+Model)\s*\}', kv_block[pos:])
    if m:
        return m.group(1)  # e.g. "CortexA57Model"
    return None


def patch(build_dir: str, keep_cpu: str) -> None:
    inc = os.path.join(build_dir, INC_PATH)
    if not os.path.exists(inc):
        print(f"ERROR: {inc} not found — is BUILD_DIR correct?")
        sys.exit(1)

    with open(inc) as f:
        content = f.read()

    if SENTINEL in content:
        print("Already patched — nothing to do.")
        return

    # Resolve which model struct to keep (if any).
    keep_model: str | None = None
    if keep_cpu:
        keep_model = find_keep_model(content, keep_cpu)
        if keep_model is None:
            print(f"WARNING: {keep_cpu!r} not found in AArch64SubTypeKV — "
                  f"stripping all models.")
        else:
            print(f"Keeping scheduling model for {keep_cpu!r}: {keep_model}")

    keep_proc  = (keep_model + "ProcResources") if keep_model else None
    keep_sched = (keep_model + "SchedClasses")  if keep_model else None

    n_stripped_proc  = 0
    n_stripped_sched = 0

    # Step 1 — replace  "  XxxModelProcResources,"  with  nullptr.
    # Only matches struct-initializer lines (2-space indent, identifier ending
    # in ModelProcResources, trailing comma).  Does NOT match array definitions
    # ("static const ... XxxModelProcResources[] = {").
    def replace_proc(m: re.Match) -> str:
        nonlocal n_stripped_proc
        if keep_proc and m.group(2) == keep_proc:
            return m.group(0)
        n_stripped_proc += 1
        return f'{m.group(1)}nullptr,  {SENTINEL}'

    content = re.sub(
        r'^(  )(\w+ModelProcResources),[ \t]*$',
        replace_proc,
        content,
        flags=re.MULTILINE,
    )

    # Step 2 — same for SchedClasses.
    def replace_sched(m: re.Match) -> str:
        nonlocal n_stripped_sched
        if keep_sched and m.group(2) == keep_sched:
            return m.group(0)
        n_stripped_sched += 1
        return f'{m.group(1)}nullptr,  {SENTINEL}'

    content = re.sub(
        r'^(  )(\w+ModelSchedClasses),[ \t]*$',
        replace_sched,
        content,
        flags=re.MULTILINE,
    )

    # Step 3 — zero NumProcResourceKinds and NumSchedClasses for each stripped
    # pair.  Pattern after steps 1+2 (for a stripped model):
    #   nullptr,  // EJIT: stripped        ← ProcResources
    #   nullptr,  // EJIT: stripped        ← SchedClasses   ← matched here
    #   NN,                                ← NumProcResourceKinds
    #   MM,                                ← NumSchedClasses
    content, n_counts = re.subn(
        r'(  nullptr,  ' + re.escape(SENTINEL) + r'\n)'
        r'  (\d+),\n'
        r'  (\d+),\n',
        r'\1'
        r'  0,  ' + SENTINEL + r' (NumProcResourceKinds)\n'
        r'  0,  ' + SENTINEL + r' (NumSchedClasses)\n',
        content,
    )

    with open(inc, "w") as f:
        f.write(content)

    kept_note = f" (kept {keep_model})" if keep_model else " (all stripped)"
    print(f"Patched{kept_note}:")
    print(f"  {n_stripped_proc}  ProcResources references  → nullptr")
    print(f"  {n_stripped_sched}  SchedClasses references   → nullptr")
    print(f"  {n_counts}  count-field pairs         → 0, 0")
    est_kb = n_stripped_proc * 24836 // 1024
    print()
    print("Next steps:")
    print(f"  ninja -C {build_dir} LLVMAArch64CodeGen LLVMAArch64Desc")
    print("  (then re-run run_aarch64_pipeline.sh)")
    print()
    print(f"Expected savings in ejit.o: ~{est_kb} KB ({n_stripped_proc} models stripped)")


if __name__ == "__main__":
    build_dir = sys.argv[1] if len(sys.argv) > 1 else "build_release_aarch64"
    keep_cpu  = sys.argv[2] if len(sys.argv) > 2 else ""
    patch(build_dir, keep_cpu)
