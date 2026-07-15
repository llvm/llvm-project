#!/usr/bin/env python3
"""
Generate C test files that call ACLE builtins found in a JSON manifest.

Expected JSON input format (array of objects):
[
  {
    "guard": "sve,(sve2p1|sme)",
    "streaming_guard": "sme",
    "flags": "feature-dependent",
    "builtin": "svint16_t svrevd_s16_z(svbool_t, svint16_t);"
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

assert sys.version_info >= (3, 7), "Only Python 3.7+ is supported."


# Are we testing arm_sve.h or arm_sme.h based builtins.
class Mode(Enum):
    SVE = "sve"
    SME = "sme"


class FunctionType(Enum):
    NORMAL = "normal"
    STREAMING = "streaming"
    STREAMING_COMPATIBLE = "streaming-compatible"


# Builtins are grouped by their required features.
@dataclass(frozen=True, order=True)
class BuiltinContext:
    guard: str
    streaming_guard: str
    flags: Tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"// Properties: "
            f'guard="{self.guard}" '
            f'streaming_guard="{self.streaming_guard}" '
            f'flags="{",".join(self.flags)}"'
        )

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "BuiltinContext":
        flags = tuple(p.strip() for p in obj["flags"].split(",") if p.strip())
        return cls(obj["guard"], obj["streaming_guard"], flags)


# --- Parsing builtins -------------------------------------------------------

# Captures the full function *declaration* inside the builtin string, e.g.:
#   "svint16_t svrevd_s16_z(svbool_t, svint16_t);"
# group(1) => "svint16_t svrevd_s16_z"
# group(2) => "svbool_t, svint16_t"
FUNC_RE = re.compile(r"^\s*([a-zA-Z_][\w\s\*]*[\w\*])\s*\(\s*([^)]*)\s*\)\s*;\s*$")

# Pulls the final word out of the left side (the function name).
NAME_RE = re.compile(r"([a-zA-Z_][\w]*)\s*$")


def parse_builtin_declaration(decl: str) -> Tuple[str, List[str]]:
    """Return (func_name, param_types) from a builtin declaration string.

    Example:
      decl = "svint16_t svrevd_s16_z(svbool_t, svint16_t);"
      => ("svrevd_s16_z", ["svbool_t", "svint16_t"])
    """
    m = FUNC_RE.match(decl)
    if not m:
        raise ValueError(f"Unrecognized builtin declaration syntax: {decl!r}")

    left = m.group(1)  # return type + name
    params = m.group(2).strip()

    name_m = NAME_RE.search(left)
    if not name_m:
        raise ValueError(f"Could not find function name in: {decl!r}")
    func_name = name_m.group(1)

    param_types: List[str] = []
    if params:
        # Split by commas respecting no pointers/arrays with commas (not expected here)
        param_types = [p.strip() for p in params.split(",") if p.strip()]

    return func_name, param_types


# --- Variable synthesis -----------------------------------------------------

# Pick a safe (ideally non-zero) value for literal types
LITERAL_TYPES_MAP: dict[str, str] = {
    "ImmCheck0_0": "0",
    "ImmCheck0_1": "1",
    "ImmCheck0_2": "2",
    "ImmCheck0_3": "2",
    "ImmCheck0_7": "2",
    "ImmCheck0_13": "2",
    "ImmCheck0_15": "2",
    "ImmCheck0_31": "2",
    "ImmCheck0_63": "2",
    "ImmCheck0_255": "2",
    "ImmCheck1_1": "1",
    "ImmCheck1_3": "2",
    "ImmCheck1_7": "2",
    "ImmCheck1_16": "2",
    "ImmCheck1_32": "2",
    "ImmCheck1_64": "2",
    "ImmCheck2_4_Mul2": "2",
    "ImmCheckComplexRot90_270": "90",
    "ImmCheckComplexRotAll90": "90",
    "ImmCheckCvt": "2",
    "ImmCheckExtract": "2",
    "ImmCheckLaneIndexCompRotate": "1",
    "ImmCheckLaneIndexDot": "1",
    "ImmCheckLaneIndex": "1",
    "ImmCheckShiftLeft": "2",
    "ImmCheckShiftRightNarrow": "2",
    "ImmCheckShiftRight": "2",
    "enum svpattern": "SV_MUL3",
    "enum svprfop": "SV_PSTL1KEEP",
    "void": "",
}


def make_arg_for_type(ty: str) -> Tuple[str, str]:
    """Return (var_decl, var_use) for a parameter type.

    Literal types return an empty declaration and a value that will be accepted
    by clang's semantic literal validation.
    """
    # Compress whitespace and remove non-relevant qualifiers.
    ty = re.sub(r"\s+", " ", ty.strip()).replace(" const", "")
    if ty in LITERAL_TYPES_MAP:
        return "", LITERAL_TYPES_MAP[ty]

    if ty.startswith("ImmCheck") or ty.startswith("enum "):
        print(f"Failed to parse potential literal type: {ty}", file=sys.stderr)

    name = ty.replace(" ", "_").replace("*", "ptr") + "_val"
    return f"{ty} {name};", name


# NOTE: Parsing is limited to the minimum required for guard strings.
# Specifically the expected input is of the form:
#   feat1,feat2,...(feat3 | feat4 | ...),...
def expand_feature_guard(
    guard: str, flags: Sequence[str], base_feature: str = ""
) -> list[set[str]]:
    """
    Expand a guard expression where ',' = AND and '|' = OR, with parentheses
    grouping OR-expressions. Returns a list of feature sets.
    """
    if not guard:
        return []

    parts = re.split(r",(?![^(]*\))", guard)

    choices_per_part = []
    for part in parts:
        if part.startswith("(") and part.endswith(")"):
            choices_per_part.append(part[1:-1].split("|"))
        else:
            choices_per_part.append([part])

    # Add feature that is common to all
    if base_feature:
        choices_per_part.append([base_feature])

    if "requires-zt" in flags:
        choices_per_part.append(["sme2"])

    # construct list of feature sets
    results = []
    for choice in product(*choices_per_part):
        choice_set = set(choice)
        results.append(choice_set)

    # remove superset and duplicates
    unique = []
    for s in results:
        if any(s > other for other in results):
            continue
        if s not in unique:
            unique.append(s)

    return unique


def cc1_args_for_features(features: set[str]) -> str:
    return " ".join("-target-feature +" + s for s in sorted(features))


def sanitise_guard(s: str) -> str:
    """Rewrite guard strings in a form more suitable for file naming."""
    replacements = {
        ",": "_AND_",
        "|": "_OR_",
        "(": "_LP_",
        ")": "_RP_",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def make_filename(prefix: str, ctx: BuiltinContext, ext: str) -> str:
    parts = [sanitise_guard(ctx.guard), sanitise_guard(ctx.streaming_guard)]
    sanitised_guard = "___".join(p for p in parts if p)

    if "streaming-only" in ctx.flags:
        prefix += "_streaming_only"
    elif "streaming-compatible" in ctx.flags:
        prefix += "_streaming_compatible"
    elif "feature-dependent" in ctx.flags:
        prefix += "_feature_dependent"
    else:
        prefix += "_non_streaming_only"

    return f"{prefix}_{sanitised_guard}{ext}"


# --- Code Generation --------------------------------------------------------


def emit_streaming_guard_run_lines(ctx: BuiltinContext) -> str:
    """Emit lit RUN lines that will exercise the relevant Sema diagnistics."""
    run_prefix = "// RUN: %clang_cc1 %s -fsyntax-only -triple aarch64-none-linux-gnu"
    out: List[str] = []

    # All RUN lines have SVE and SME enabled
    guard_features = expand_feature_guard(ctx.guard, ctx.flags, "sme")
    streaming_guard_features = expand_feature_guard(
        ctx.streaming_guard, ctx.flags, "sve"
    )

    if "streaming-only" in ctx.flags:
        assert not guard_features
        # Generate RUN lines for features only available to streaming functions
        for feats in streaming_guard_features:
            out.append(
                f"{run_prefix} {cc1_args_for_features(feats)} -verify=streaming-guard"
            )
    elif "streaming-compatible" in ctx.flags:
        assert not guard_features
        # NOTE: Streaming compatible builtins don't require SVE.
        # Generate RUN lines for features available to all functions.
        for feats in expand_feature_guard(ctx.streaming_guard, ctx.flags):
            out.append(f"{run_prefix} {cc1_args_for_features(feats)} -verify")
        out.append("// expected-no-diagnostics")
    elif "feature-dependent" in ctx.flags:
        assert guard_features and streaming_guard_features
        combined_features = expand_feature_guard(
            ctx.guard + "," + ctx.streaming_guard, ctx.flags
        )

        # Generate RUN lines for features only available to normal functions
        for feats in guard_features:
            if feats not in combined_features:
                out.append(f"{run_prefix} {cc1_args_for_features(feats)} -verify=guard")

        # Geneate RUN lines for features only available to streaming functions
        for feats in streaming_guard_features:
            if feats not in combined_features:
                out.append(
                    f"{run_prefix} {cc1_args_for_features(feats)} -verify=streaming-guard"
                )

        # Generate RUN lines for features available to all functions
        for feats in combined_features:
            out.append(f"{run_prefix} {cc1_args_for_features(feats)} -verify")

        out.append("// expected-no-diagnostics")
    else:
        assert not streaming_guard_features
        # Geneate RUN lines for features only available to normal functions
        for feats in guard_features:
            out.append(f"{run_prefix} {cc1_args_for_features(feats)} -verify=guard")

    return "\n".join(out)


def emit_streaming_guard_function(
    ctx: BuiltinContext,
    var_decls: Sequence[str],
    calls: Sequence[str],
    func_name: str,
    func_type: FunctionType = FunctionType.NORMAL,
) -> str:
    """Emit a C function calling all builtins.

    `calls` is a sequence of tuples: (name, call_line)
    """
    # Expected Sema diagnostics for invalid usage
    require_diagnostic = require_streaming_diagnostic = False
    if "streaming-only" in ctx.flags:
        if func_type != FunctionType.STREAMING:
            require_streaming_diagnostic = True
    elif "streaming-compatible" in ctx.flags:
        pass  # streaming compatible builtins are always available
    elif "feature-dependent" in ctx.flags:
        guard_features = expand_feature_guard(ctx.guard, ctx.flags, "sme")
        streaming_guard_features = expand_feature_guard(
            ctx.streaming_guard, ctx.flags, "sve"
        )
        combined_features = expand_feature_guard(
            ctx.guard + "," + ctx.streaming_guard, ctx.flags
        )

        if func_type != FunctionType.NORMAL:
            if any(feats not in combined_features for feats in guard_features):
                require_diagnostic = True
        if func_type != FunctionType.STREAMING:
            if any(
                feats not in combined_features for feats in streaming_guard_features
            ):
                require_streaming_diagnostic = True
    else:
        if func_type != FunctionType.NORMAL:
            require_diagnostic = True

    out: List[str] = []

    # Emit test function declaration
    attr: list[str] = []
    if func_type == FunctionType.STREAMING:
        attr.append("__arm_streaming")
    elif func_type == FunctionType.STREAMING_COMPATIBLE:
        attr.append("__arm_streaming_compatible")

    if "requires-za" in ctx.flags:
        attr.append('__arm_inout("za")')
    if "requires-zt" in ctx.flags:
        attr.append('__arm_inout("zt0")')
    out.append(f"void {func_name}(void) " + " ".join(attr) + "{")

    # Emit variable declarations
    for v in var_decls:
        out.append(f"  {v}")
    if var_decls:
        out.append("")

    # Emit calls
    for call in calls:
        if require_diagnostic and require_streaming_diagnostic:
            out.append(
                "  // guard-error@+2 {{builtin can only be called from a non-streaming function}}"
            )
            out.append(
                "  // streaming-guard-error@+1 {{builtin can only be called from a streaming function}}"
            )
        elif require_diagnostic:
            out.append(
                "  // guard-error@+1 {{builtin can only be called from a non-streaming function}}"
            )
        elif require_streaming_diagnostic:
            out.append(
                "  // streaming-guard-error@+1 {{builtin can only be called from a streaming function}}"
            )
        out.append(f"  {call}")

    out.append("}")
    return "\n".join(out) + "\n"


def natural_key(s: str):
    """Allow sorting akin to "sort -V"""
    return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", s)]


def build_calls_for_group(builtins: Iterable[str]) -> Tuple[List[str], List[str]]:
    """From a list of builtin declaration strings, produce:
    - a sorted list of unique variable declarations
    - a sorted list of builtin calls
    """
    var_decls: List[str] = []
    seen_types: set[str] = set()
    calls: List[str] = []

    for decl in builtins:
        fn, param_types = parse_builtin_declaration(decl)

        arg_names: List[str] = []
        for i, ptype in enumerate(param_types):
            vdecl, vname = make_arg_for_type(ptype)
            if vdecl and vdecl not in seen_types:
                seen_types.add(vdecl)
                var_decls.append(vdecl)
            arg_names.append(vname)

        calls.append(f"{fn}(" + ", ".join(arg_names) + ");")

    # Natural sort (e.g. int8_t before int16_t)
    calls.sort(key=natural_key)
    var_decls.sort(key=natural_key)

    return var_decls, calls


def gen_streaming_guard_tests(mode: Mode, json_path: Path, out_dir: Path) -> None:
    """Generate a set of Clang Sema test files to ensure SVE/SME builtins are
    callable based on the function type, or the required diagnostic is emitted.
    """
    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON {json_path}: {e}", file=sys.stderr)
        return

    # Group by (guard, streaming_guard)
    by_guard: Dict[BuiltinContext, List[str]] = defaultdict(list)
    for obj in data:
        by_guard[BuiltinContext.from_json(obj)].append(obj["builtin"])

    # For each guard pair, emit 3 functions
    for builtin_ctx, builtin_decls in by_guard.items():
        var_decls, calls = build_calls_for_group(builtin_decls)

        out_parts: List[str] = []
        out_parts.append(
            "// NOTE: File has been autogenerated by utils/aarch64_builtins_test_generator.py"
        )
        out_parts.append(emit_streaming_guard_run_lines(builtin_ctx))
        out_parts.append("")
        out_parts.append("// REQUIRES: aarch64-registered-target")
        out_parts.append("")
        out_parts.append(f"#include <arm_{mode.value}.h>")
        out_parts.append("")
        out_parts.append(str(builtin_ctx))
        out_parts.append("")
        out_parts.append(
            emit_streaming_guard_function(builtin_ctx, var_decls, calls, "test")
        )
        out_parts.append(
            emit_streaming_guard_function(
                builtin_ctx, var_decls, calls, "test_streaming", FunctionType.STREAMING
            )
        )
        out_parts.append(
            emit_streaming_guard_function(
                builtin_ctx,
                var_decls,
                calls,
                "test_streaming_compatible",
                FunctionType.STREAMING_COMPATIBLE,
            )
        )

        output = "\n".join(out_parts).rstrip() + "\n"

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = make_filename(f"arm_{mode.value}", builtin_ctx, ".c")
            (out_dir / filename).write_text(output)
        else:
            print(output)

    return


# --- Main -------------------------------------------------------------------


def existing_file(path: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"{p} is not a valid file")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit C tests for SVE/SME builtins")
    ap.add_argument(
        "json", type=existing_file, help="Path to json formatted builtin descriptions"
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="Output directory (default: stdout)"
    )
    ap.add_argument(
        "--gen-streaming-guard-tests",
        action="store_true",
        help="Generate C tests to validate SVE/SME builtin usage base on streaming attribute",
    )
    ap.add_argument(
        "--gen-target-guard-tests",
        action="store_true",
        help="Generate C tests to validate SVE/SME builtin usage based on target features",
    )
    ap.add_argument(
        "--gen-builtin-tests",
        action="store_true",
        help="Generate C tests to exercise SVE/SME builtins",
    )
    ap.add_argument(
        "--base-target-feature",
        choices=["sve", "sme"],
        help="Force builtin source (sve: arm_sve.h, sme: arm_sme.h)",
    )

    args = ap.parse_args(argv)

    # When not forced, try to infer the mode from the input, defaulting to SVE.
    if args.base_target_feature:
        mode = Mode(args.base_target_feature)
    elif args.json and args.json.name == "arm_sme_builtins.json":
        mode = Mode.SME
    else:
        mode = Mode.SVE

    # Generate test file
    if args.gen_streaming_guard_tests:
        gen_streaming_guard_tests(mode, args.json, args.out_dir)
    if args.gen_target_guard_tests:
        ap.error("--gen-target-guard-tests not implemented yet!")
    if args.gen_builtin_tests:
        ap.error("--gen-builtin-tests not implemented yet!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
