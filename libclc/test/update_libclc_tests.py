#!/usr/bin/env python3
"""A utility to wrap utils/update_cc_test_checks.py for updating CHECK lines in
libclc .cl tests for a given architecture.

The script accepts an architecture argument to determine the triple and check
prefix. Supported arch values: amdgpu, amdgcn, nvptx64, spirv, spirv64.

The script does 3 things:
1. Replaces %libclc_target and %check_prefix in the .cl file for the arch.
2. Runs update_cc_test_checks.py to update CHECK lines.
3. Reverts the .cl file back to using %libclc_target and %check_prefix.

Usage:

% libclc/test/update_libclc_tests.py amdgpu

"""

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARCH_TO_TRIPLE = {
    "amdgpu":   "amdgcn-amd-amdhsa-llvm",
    "amdgcn":   "amdgcn-amd-amdhsa-llvm",
    "nvptx64":  "nvptx64-nvidia-cuda",
    "spirv":    "spirv-unknown-mesa3d",
    "spirv64":  "spirv64-unknown-mesa3d",
}

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
UPDATE_SCRIPT = REPO_ROOT / "llvm" / "utils" / "update_cc_test_checks.py"
CLANG = REPO_ROOT / "build" / "bin" / "clang"


def find_cl_files(test_dir: Path):
    return list(test_dir.rglob("*.cl"))


def replace_in_file(path: Path, triple: str, check_prefix: str):
    content = path.read_bytes()
    content = content.replace(b"%libclc_target", triple.encode())
    content = content.replace(b"%check_prefix", check_prefix.encode())
    path.write_bytes(content)


def revert_in_file(path: Path, triple: str, check_prefix: str):
    # Only revert in the RUN line context, not in generated CHECK lines.
    content = path.read_bytes()
    content = content.replace(
        f"-target {triple}".encode(), b"-target %libclc_target"
    )
    content = content.replace(
        f"--check-prefix={check_prefix}".encode(), b"--check-prefix=%check_prefix"
    )
    path.write_bytes(content)


def file_requires_feature(path: Path, feature: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    for line in text.splitlines():
        stripped = line.strip().lstrip("//").strip()
        if stripped.startswith("REQUIRES:"):
            rest = stripped[len("REQUIRES:"):]
            features = [f.strip() for f in re.split(r",|\|\|", rest)]
            if feature in features:
                return True
    return False


def process_file(cl_file: Path, triple: str, check_prefix: str) -> bool:
    replace_in_file(cl_file, triple, check_prefix)
    cmd = [
        sys.executable,
        str(UPDATE_SCRIPT),
        "--clang", str(CLANG),
        str(cl_file),
    ]
    print(f"  update: {cl_file.relative_to(REPO_ROOT)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = result.returncode == 0
    if not ok:
        print(f"  FAILED: {result.stderr.strip()}", file=sys.stderr)
    revert_in_file(cl_file, triple, check_prefix)
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Update libclc FileCheck assertions for a given arch."
    )
    parser.add_argument(
        "arch",
        choices=list(ARCH_TO_TRIPLE.keys()),
        help="Target arch: amdgpu, amdgcn, nvptx64, spirv, spirv64",
    )
    args = parser.parse_args()

    arch = args.arch.lower()
    triple = ARCH_TO_TRIPLE[arch]
    # check_prefix matches REQUIRES feature: uppercase of canonical arch name
    # amdgpu -> AMDGCN (same triple as amdgcn), others uppercased
    if arch == "amdgpu":
        check_prefix = "AMDGCN"
    else:
        check_prefix = arch.upper()

    cl_files = find_cl_files(SCRIPT_DIR)
    # Only process files that REQUIRES this feature
    target_files = [f for f in cl_files if file_requires_feature(f, check_prefix)]

    if not target_files:
        print(f"No .cl files found with REQUIRES: {check_prefix}")
        return

    print(f"arch={arch}  triple={triple}  check_prefix={check_prefix}")
    print(f"Processing {len(target_files)} file(s)...")

    failed = []
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, f, triple, check_prefix): f
            for f in target_files
        }
        for future in as_completed(futures):
            if not future.result():
                failed.append(futures[future])

    if failed:
        print(f"\n{len(failed)} file(s) failed:", file=sys.stderr)
        for f in failed:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Done. Updated {len(target_files)} file(s).")


if __name__ == "__main__":
    main()
