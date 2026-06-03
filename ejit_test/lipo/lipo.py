#!/usr/bin/env python3
"""
EJIT Lipo — Extract the minimal set of .o files from LLVM .a archives.

Three-stage pipeline to produce a single ejit.o from ~36 LLVM .a files:

  extract    Compile reference binary → linker map → nm -u dependency tracing
             → single .a with only the .o files actually needed.
  gc-merge   ld -r --gc-sections on the extracted .a, rooted at EJIT API
             entry points.  Also strips ARM $x/$d mapping symbols and .group
             metadata when llvm-objcopy is available.
  merge      ld -r -T merge.ld → single relocatable ejit.o with merged
             .text/.rodata/.data sections.

Usage:
  python3 lipo.py extract  --arch=x86|aarch64 --build-dir=PATH [--output=PATH]
  python3 lipo.py gc-merge --input=PATH --build-dir=PATH [--output=PATH]
  python3 lipo.py merge    --input=PATH --build-dir=PATH [--output=PATH]

Default compiler/linker are build-dir/bin/clang++ and build-dir/bin/ld.lld.
Override with --cxx / --ld for cross-compilation.

The resulting ejit.o (~30-40 MB) can replace all individual LLVM .a files
when linking EJIT test binaries.
"""

import subprocess as sp, os, sys, re, argparse, struct, glob


# ── per-architecture configuration ──────────────────────────────────────────

TARGET_LIBS = {
    "x86": [
        "libLLVMX86CodeGen.a", "libLLVMX86Desc.a", "libLLVMX86Info.a",
    ],
    "aarch64": [
        "libLLVMAArch64CodeGen.a", "libLLVMAArch64Desc.a",
        "libLLVMAArch64Info.a", "libLLVMAArch64Utils.a",
    ],
}

COMMON_LIBS = [
    "libLLVMCore.a", "libLLVMSupport.a", "libLLVMDemangle.a",
    "libLLVMBinaryFormat.a", "libLLVMBitReader.a", "libLLVMBitstreamReader.a",
    "libLLVMAnalysis.a", "libLLVMScalarOpts.a", "libLLVMInstCombine.a",
    "libLLVMipo.a", "libLLVMTransformUtils.a", "libLLVMCodeGen.a",
    "libLLVMCodeGenTypes.a", "libLLVMTarget.a", "libLLVMTargetParser.a",
    "libLLVMSelectionDAG.a", "libLLVMAsmPrinter.a", "libLLVMMC.a",
    "libLLVMObject.a", "libLLVMProfileData.a", "libLLVMExecutionEngine.a",
    "libLLVMOrcJIT.a", "libLLVMOrcShared.a", "libLLVMJITLink.a",
    "libLLVMRemarks.a", "libLLVMOption.a", "libLLVMMCDisassembler.a",
    "libLLVMIRPrinter.a",
    "libLLVMOrcTargetProcess.a", "libLLVMRuntimeDyld.a", "libLLVMBitWriter.a",
    "libLLVMGlobalISel.a",
]


def all_libs(arch):
    """Return the full list of .a basenames for the given architecture."""
    return COMMON_LIBS + TARGET_LIBS.get(arch, [])


def cxx(build_dir):
    return os.path.join(build_dir, "bin", "clang++")


def ld(build_dir):
    return os.path.join(build_dir, "bin", "ld.lld")


def lib_dir(build_dir):
    return os.path.join(build_dir, "lib")


# ── symbol index ────────────────────────────────────────────────────────────

def build_symbol_index(build_dir):
    """
    Build a mangled-name → (archive_basename, member.o) map from all
    LLVM .a files in the build directory.  Uses nm --print-armap for speed.
    """
    L = lib_dir(build_dir)
    idx = {}
    for a in sorted(glob.glob(os.path.join(L, "libLLVM*.a"))):
        aname = os.path.basename(a)
        r = sp.run(["nm", "--print-armap", a], capture_output=True, text=True)
        for line in r.stdout.split("\n"):
            if " in " in line:
                mangled, member = line.split(" in ", 1)
                idx[mangled.strip()] = (aname, member.strip())
    return idx


# ── objcopy helpers ──────────────────────────────────────────────────────────

def _find_objcopy(build_dir):
    """Return (tool_path, is_llvm) for the best available objcopy."""
    llvm_oc = os.path.join(build_dir, "bin", "llvm-objcopy")
    if os.path.exists(llvm_oc):
        return llvm_oc, True
    # GNU objcopy struggles with extended ELF (>65280 sections) from ld -r.
    # Try it anyway; callers handle failure gracefully.
    return "objcopy", False


def _try_strip_arm_mapping_symbols(merged_o, work_dir, build_dir):
    """Strip ARM $x/$d mapping symbols from *merged_o* (best-effort).

    $x (code) and $d (data) are ARM ELF mapping symbols inserted by the
    assembler.  They are not needed after a partial link and inflate the
    symtab by ~60 000 entries on aarch64.  Failure is non-fatal: the
    symbols are harmless metadata.
    """
    objcopy_tool, is_llvm = _find_objcopy(build_dir)
    nostrip_o = os.path.join(work_dir, "_nostrip.o")

    r = sp.run([objcopy_tool, "-w", "-N", "$x", "-N", "$d",
                merged_o, nostrip_o], capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(nostrip_o):
        try:
            before = len(sp.run(["nm", merged_o], capture_output=True,
                                text=True).stdout.splitlines())
            after = len(sp.run(["nm", nostrip_o], capture_output=True,
                               text=True).stdout.splitlines())
            if after < before:
                print(f"       stripped {before - after} $x/$d mapping symbols"
                      f"{' (llvm-objcopy)' if is_llvm else ''}")
                # Replace merged_o with the stripped version
                os.replace(nostrip_o, merged_o)
                return
            os.unlink(nostrip_o)
        except OSError:
            if os.path.exists(nostrip_o):
                os.unlink(nostrip_o)
    elif is_llvm:
        print(f"       note: llvm-objcopy could not strip $x/$d (non-fatal)")
    else:
        print(f"       note: GNU objcopy cannot handle this ELF ("
              f"{'>65280' if True else ''}sections from ld -r)."
              f"  Build llvm-objcopy to enable $x/$d stripping."
              f"  The $x/$d symbols are harmless ARM mapping metadata.")


def _try_remove_group(merged_o, nogroup_o, build_dir):
    """Remove .group (COMDAT) section from *merged_o* (best-effort).

    After ld -r --gc-sections, COMDAT .group metadata is no longer needed.
    merge.ld also discards .group, so failure here is non-fatal.
    """
    objcopy_tool, is_llvm = _find_objcopy(build_dir)
    r = sp.run([objcopy_tool, "--remove-section=.group", merged_o, nogroup_o],
               capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(nogroup_o):
        before_mb = os.path.getsize(merged_o) / (1024 * 1024)
        after_mb = os.path.getsize(nogroup_o) / (1024 * 1024)
        if after_mb < before_mb:
            print(f"       after --remove-section=.group: {after_mb:.0f} MB")
        else:
            # No size reduction; keep original
            os.unlink(nogroup_o)


# ── extract mode ────────────────────────────────────────────────────────────

def doit_extract(args):
    build_dir = os.path.abspath(args.build_dir)
    arch = args.arch
    L = lib_dir(build_dir)
    # Default output: ejit_test/lipo/ (alongside this script)
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"libejit_lipo_{arch}.a")
    output = args.output or default_out
    work = os.path.join(os.path.dirname(output), f".lipo_work_{arch}")
    os.makedirs(work, exist_ok=True)

    CXX = args.cxx or cxx(build_dir)
    LD = args.ld or ld(build_dir)
    custom_ld = args.ld is not None

    # Quick test object to drive the link
    test_o = os.path.join(work, "_test_main.o")
    sp.run([CXX, "-x", "c", "-", "-O2", "-c", "-o", test_o],
           input=b"int main(){return 0;}", capture_output=True)

    libs = all_libs(arch)
    all_a = " ".join(os.path.join(L, f) for f in libs)
    ejit_a = os.path.join(L, "libLLVMEJIT.a")

    # ── 1. Build symbol index ────────────────────────────────────────────
    print("[1/4] Building symbol index ...", flush=True)
    sym2file = build_symbol_index(build_dir)
    print(f"       {len(sym2file)} symbols indexed")

    # ── 2. Generate linker map (successful link) ──────────────────────────
    print("[2/4] Generating linker map ...", flush=True)
    if custom_ld:
        ld_dir = os.path.dirname(LD)
        fuse_ld_flag = f"-fuse-ld=lld"
        link_cmd = [CXX, f"-B{ld_dir}", fuse_ld_flag, "-L/tmp"]
    else:
        link_cmd = [CXX, f"-fuse-ld={LD}"]
    r = sp.run(link_cmd + [
        "-Os", "-Wl,--gc-sections",
        "-Wl,--print-map",
        "-Wl,--allow-multiple-definition",
        f"-Wl,--whole-archive", ejit_a, f"-Wl,--no-whole-archive",
        *all_a.split(), "-lz", "-lpthread", "-ldl", test_o,
        "-o", os.path.join(work, "_ref")
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR: reference link failed. Is the build directory valid?")
        print(r.stderr[-500:])
        sys.exit(1)
    map_text = r.stdout + r.stderr

    # ── 3. Extract .o from map + trace dependencies ───────────────────────
    print("[3/4] Extracting .o files + tracing dependencies ...", flush=True)
    if args.exclude:
        print(f"       excluding patterns: {args.exclude}")
    pattern = re.compile(r"(libLLVM\S+\.a)\(([^)]+)\)")
    seen = set()   # (aname, member)
    extracted = set()  # unique_name present in work/

    def _is_excluded(member):
        for pat in args.exclude:
            if pat in member:
                return True
        return False

    # Helper: extract member from archive with unique name
    def extract_one(aname, member):
        unique = f"{aname.replace('.a','')}__{member}"
        if unique in extracted:
            return unique
        arch = os.path.join(L, aname)
        sp.run(["ar", "x", arch, member], cwd=work, capture_output=True)
        src = os.path.join(work, member)
        dst = os.path.join(work, unique)
        if os.path.exists(src):
            os.rename(src, dst)
        extracted.add(unique)
        return unique

    # From map
    for m in pattern.finditer(map_text):
        aname, member = m.group(1), m.group(2)
        if _is_excluded(member):
            continue
        key = (aname, member)
        if key not in seen:
            seen.add(key)
            extract_one(aname, member)

    # EJIT (whole-archive)
    sp.run(["ar", "x", ejit_a], cwd=work, capture_output=True)
    for f in os.listdir(work):
        if f.endswith(".o") and not f.startswith("libLLVM"):
            src = os.path.join(work, f)
            dst = os.path.join(work, f"libLLVMEJIT__{f}")
            if os.path.exists(src):
                os.rename(src, dst)
            extracted.add(f"libLLVMEJIT__{f}")

    # Iterative dependency trace via nm -u
    for it in range(10):
        added = 0
        for unique in sorted(extracted):
            o_path = os.path.join(work, unique)
            r = sp.run(["nm", "-u", o_path], capture_output=True, text=True)
            for line in r.stdout.split("\n"):
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == "U":
                    mangled = parts[-1]
                    if mangled in sym2file:
                        aname, member = sym2file[mangled]
                        if _is_excluded(member):
                            continue
                        key = (aname, member)
                        if key not in seen:
                            seen.add(key)
                            extract_one(aname, member)
                            added += 1
        if added == 0:
            break
        print(f"       iteration {it+1}: +{added} .o", flush=True)

    # ── 4. Build single .a ────────────────────────────────────────────────
    print(f"[4/4] Building {output} ...", flush=True)
    o_files = [os.path.join(work, f) for f in sorted(os.listdir(work))
               if f.endswith(".o")]
    # Remove old archive to avoid stale members (ar crs only replaces, not deletes)
    if os.path.exists(output):
        os.unlink(output)
    sp.run(["ar", "crs", output, *o_files], capture_output=True)

    sz_mb = os.path.getsize(output) / (1024 * 1024)
    orig_mb = sum(os.path.getsize(os.path.join(L, f)) for f in libs) / (1024 * 1024)
    print(f"       {len(o_files)} .o files, {sz_mb:.0f} MB")
    print(f"       (from {len(libs)} .a = {orig_mb:.0f} MB)")
    print(f"       output: {output}")


# ── gc-merge mode ───────────────────────────────────────────────────────────

def doit_gc_merge(args):
    input_a = os.path.abspath(args.input)
    output = args.output or os.path.join(os.path.dirname(input_a),
                       os.path.basename(input_a).replace(".a", "_gc.a"))
    work = os.path.join(os.path.dirname(output), ".lipo_gc_work")
    # Clean up any leftover files from previous runs
    if os.path.isdir(work):
        for f in os.listdir(work):
            os.unlink(os.path.join(work, f))
    os.makedirs(work, exist_ok=True)

    LD = args.ld or ld(args.build_dir)
    CXX = cxx(args.build_dir)

    # ── 1. Extract .o from input .a ───────────────────────────────────────
    print("[1/3] Extracting .o from input .a ...", flush=True)
    sp.run(["ar", "x", input_a], cwd=work, capture_output=True)
    o_files = [os.path.join(work, f) for f in sorted(os.listdir(work))
               if f.endswith(".o")]
    before_mb = sum(os.path.getsize(o) for o in o_files) / (1024 * 1024)
    print(f"       {len(o_files)} .o files, {before_mb:.0f} MB")

    # ── 2. ld -r --gc-sections ────────────────────────────────────────────
    print("[2/3] Running ld -r --gc-sections ...", flush=True)

    # EJIT API entry points as gc roots
    ejit_api = [
        "ejit_init", "ejit_shutdown", "ejit_activate", "ejit_deactivate",
        "ejit_activate_all", "ejit_deactivate_all", "ejit_is_active",
        "ejit_activate_array", "ejit_deactivate_array", "ejit_get_stats",
        "ejit_register_symbol", "ejit_register_bitcode",
        "ejit_register_period_array", "ejit_register_static_var",
        "ejit_clear_cache", "ejit_compile_or_get", "ejit_invalidate",
        "ejit_set_compile_mode", "ejit_get_compile_mode", "ejit_get_last_error",
    ]
    u_flags = []
    for s in ejit_api:
        u_flags.extend(["-u", s])

    merged_o = os.path.join(work, "_merged.o")
    r = sp.run(
        [LD, "-r", "-o", merged_o, "--gc-sections", "--entry=ejit_init",
         "--allow-multiple-definition"]
        + u_flags
        + o_files,
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("ERROR: ld -r failed")
        print(r.stderr[:500])
        sys.exit(1)

    after_mb = os.path.getsize(merged_o) / (1024 * 1024)
    print(f"       {before_mb:.0f} MB -> {after_mb:.0f} MB (gc-sections)")

    # ── 2b. Strip ARM $x/$d mapping symbols (metadata, not needed after link) ──
    # $x/$d are ARM mapping symbols (~60K in aarch64) that only help
    # disassemblers; they are safe to strip.  Prefer llvm-objcopy (handles
    # extended ELF with >65280 sections); GNU objcopy rejects such files.
    _try_strip_arm_mapping_symbols(merged_o, work, args.build_dir)

    # ── 2c. Remove .group (COMDAT metadata, not needed after partial link) ──
    # Note: merge.ld also discards .group, so this is a best-effort early clean.
    nogroup_o = os.path.join(work, "_nogroup.o")
    _try_remove_group(merged_o, nogroup_o, args.build_dir)
    if os.path.exists(nogroup_o):
        merged_o = nogroup_o

    # ── 3. Build new .a ───────────────────────────────────────────────────
    print(f"[3/3] Building {output} ...", flush=True)
    sp.run(["ar", "crs", output, merged_o], capture_output=True)
    sz_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"       {sz_mb:.0f} MB")
    print(f"       output: {output}")


# ── merge mode ──────────────────────────────────────────────────────────────

def doit_merge(args):
    """ld -r with merge.ld to produce a single compact .o (ejit.o)."""
    build_dir = os.path.abspath(args.build_dir)
    input_a = os.path.abspath(args.input)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    merge_ld = os.path.join(script_dir, "merge.ld")

    if not os.path.exists(merge_ld):
        print(f"ERROR: merge.ld not found at {merge_ld}")
        sys.exit(1)

    output = args.output or os.path.join(script_dir, "ejit.o")
    LD = args.ld or ld(build_dir)

    print(f"[merge] ld -r -T merge.ld -> {output} ...", flush=True)
    r = sp.run([
        LD, "-r", "-o", output, "-T", merge_ld,
        "--whole-archive", input_a,
    ], capture_output=True, text=True)

    if r.returncode != 0:
        print("ERROR: ld -r failed")
        print(r.stderr[:500])
        sys.exit(1)

    sz_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"       {sz_mb:.0f} MB")
    print(f"       output: {output}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="EJIT Lipo tool")
    sub = p.add_subparsers(dest="mode", required=True)

    e = sub.add_parser("extract", help="Extract used .o -> single .a")
    e.add_argument("--arch", required=True, choices=["x86", "aarch64"])
    e.add_argument("--build-dir", required=True, help="LLVM build directory")
    e.add_argument("--output", help="Output .a path")
    e.add_argument("--cxx", help="Override C++ compiler (default: build-dir/bin/clang++)")
    e.add_argument("--ld", help="Override linker (default: build-dir/bin/ld.lld)")
    e.add_argument("--exclude", action="append", default=[],
                   help="Exclude .o files matching this substring (repeatable)")

    g = sub.add_parser("gc-merge", help="gc-merge an existing lipo .a")
    g.add_argument("--input", required=True, help="Input .a from extract step")
    g.add_argument("--build-dir", required=True, help="LLVM build directory")
    g.add_argument("--ld", help="Override linker (default: build-dir/bin/ld.lld)")
    g.add_argument("--output", help="Output .a path")

    m = sub.add_parser("merge", help="ld -r merge into single ejit.o")
    m.add_argument("--input", required=True, help="Input .a from gc-merge step")
    m.add_argument("--build-dir", required=True, help="LLVM build directory")
    m.add_argument("--ld", help="Override linker (default: build-dir/bin/ld.lld)")
    m.add_argument("--output", help="Output .o path (default: ejit.o alongside lipo.py)")

    args = p.parse_args()

    if args.mode == "extract":
        doit_extract(args)
    elif args.mode == "gc-merge":
        doit_gc_merge(args)
    elif args.mode == "merge":
        doit_merge(args)


if __name__ == "__main__":
    main()
