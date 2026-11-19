#!/usr/bin/env python3
#
# ===- libc_size_check.py - Binary size checks for LLVM libc ---*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#
"""Binary size checks for LLVM libc.

The `objects` command reads the symbol table (``llvm-nm -S -C``) and the
section sizes (``llvm-size -A``) of each entrypoint object listed in a manifest
and evaluates the rules from a JSON rules file against them. Rules are selected
by the active ``LIBC_CONF_MATH_OPTIMIZATIONS`` flags and by the target's
predefined macros, so the same rules file can be used for every ELF target and
configuration.

The `probes` command links minimal programs that call a few functions against
the libc archives and reports how much code and data they pull in.

See README.md in this directory for details.
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

# The default regular expression matching the versioned libc namespace prefix
# of a demangled symbol, e.g. "__llvm_libc_24_0_0_git::".
DEFAULT_SYMBOL_PREFIX_REGEX = r"^__llvm_libc[0-9A-Za-z_]*::"

# Composite LIBC_MATH_* flags and the individual flags they stand for. Keep in
# sync with libc/src/__support/macros/optimization.h.
COMPOSITE_MATH_FLAGS = {
    "LIBC_MATH_FAST": [
        "LIBC_MATH_SKIP_ACCURATE_PASS",
        "LIBC_MATH_SMALL_TABLES",
        "LIBC_MATH_NO_ERRNO",
        "LIBC_MATH_NO_EXCEPT",
        "LIBC_MATH_ASSUME_ROUND_NEAREST_ONLY",
    ],
}

# nm symbol types that denote named data objects (tables, constants, writable
# variables). Lower case letters are local symbols.
DATA_SYMBOL_TYPES = set("VvRrDdBbGgSsu")

# Section name prefixes that hold writable data.
WRITABLE_SECTION_PREFIXES = (".data", ".bss", ".sdata", ".sbss", ".tdata", ".tbss")

STATUS_FAIL = "fail"
STATUS_KNOWN = "known"
STATUS_NOTE = "note"


class Symbol:
    """A defined symbol of an object file."""

    def __init__(self, name: str, raw_name: str, sym_type: str, size: int):
        self.name = name  # Demangled, with the libc namespace prefix removed.
        self.raw_name = raw_name  # Demangled as printed by llvm-nm.
        self.type = sym_type
        self.size = size

    def is_data(self) -> bool:
        return self.type in DATA_SYMBOL_TYPES and not self.raw_name.startswith(".")

    def __repr__(self):
        return "Symbol(%r, %r, %d)" % (self.name, self.type, self.size)


class ObjectInfo:
    """Symbols and section sizes of one entrypoint object."""

    def __init__(
        self,
        function: str,
        paths: List[str],
        defined: List[Symbol],
        undefined: Set[str],
        sections: Dict[str, int],
        macros: Optional[Dict[str, str]] = None,
    ):
        self.function = function
        self.paths = paths
        self.defined = defined
        self.undefined = undefined
        self.sections = sections
        # Predefined macros of the options the object was compiled with.
        self.macros = macros or {}

    def data_symbols(self) -> List[Symbol]:
        return [s for s in self.defined if s.is_data()]

    def largest_data_symbol(self) -> Optional[Symbol]:
        data = self.data_symbols()
        if not data:
            return None
        return max(data, key=lambda s: (s.size, s.name))

    def writable_bytes(self, allow: Iterable["re.Pattern"] = ()) -> int:
        return sum(size for _, size in self.writable_sections(allow))

    def writable_sections(
        self, allow: Iterable["re.Pattern"] = ()
    ) -> List[Tuple[str, int]]:
        """Returns the non-empty writable sections not matched by `allow`."""
        allow = list(allow)
        return sorted(
            (name, size)
            for name, size in self.sections.items()
            if size
            and name.startswith(WRITABLE_SECTION_PREFIXES)
            and not any(a.search(name) for a in allow)
        )

    def total_bytes(self) -> int:
        """Size of all allocatable sections (code, rodata and data)."""
        total = 0
        for name, size in self.sections.items():
            if name.startswith((".text", ".rodata")) or name.startswith(
                WRITABLE_SECTION_PREFIXES
            ):
                total += size
        return total


class Finding:
    """The result of a rule violation, known exception or advisory note."""

    def __init__(self, rule_id: str, function: str, status: str, message: str):
        self.rule_id = rule_id
        self.function = function
        self.status = status
        self.message = message

    def to_json(self) -> dict:
        return {
            "rule": self.rule_id,
            "function": self.function,
            "status": self.status,
            "message": self.message,
        }

    def __repr__(self):
        return "Finding(%r, %r, %r, %r)" % (
            self.rule_id,
            self.function,
            self.status,
            self.message,
        )


# ===----------------------------------------------------------------------===#
# Parsing of tool output.
# ===----------------------------------------------------------------------===#

_NM_SIZED_RE = re.compile(r"^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+(\S)\s+(.+)$")
_NM_UNSIZED_RE = re.compile(r"^([0-9a-fA-F]+)\s+(\S)\s+(.+)$")
_NM_UNDEFINED_RE = re.compile(r"^\s*([Uwv])\s+(.+)$")


def strip_prefix(name: str, prefix_re: "re.Pattern") -> str:
    return prefix_re.sub("", name, count=1)


def parse_nm(text: str, prefix_re: "re.Pattern") -> Tuple[List[Symbol], Set[str]]:
    """Parses ``llvm-nm -S -C`` output into defined symbols and undefined names.

    Lines that name the input file (as printed for archives or multiple inputs)
    and empty lines are ignored.
    """
    defined = []
    undefined = set()
    for line in text.splitlines():
        if not line.strip() or line.rstrip().endswith(":"):
            continue
        m = _NM_UNDEFINED_RE.match(line)
        if m:
            undefined.add(m.group(2).strip())
            continue
        m = _NM_SIZED_RE.match(line)
        if m:
            raw = m.group(4).strip()
            defined.append(
                Symbol(
                    strip_prefix(raw, prefix_re), raw, m.group(3), int(m.group(2), 16)
                )
            )
            continue
        m = _NM_UNSIZED_RE.match(line)
        if m:
            raw = m.group(3).strip()
            defined.append(Symbol(strip_prefix(raw, prefix_re), raw, m.group(2), 0))
    return defined, undefined


def parse_size_sysv(text: str) -> Dict[str, int]:
    """Parses ``llvm-size -A`` output into a map of section name to size.

    Sizes of sections with the same name (e.g. COMDAT groups) are summed.
    """
    sections: Dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 3 or not parts[0].startswith("."):
            continue
        try:
            size = int(parts[1], 0)
        except ValueError:
            continue
        sections[parts[0]] = sections.get(parts[0], 0) + size
    return sections


def parse_math_flags(value: str) -> Set[str]:
    """Returns the set of LIBC_MATH_* flags named in a configuration value.

    ``value`` is the value of LIBC_CONF_MATH_OPTIMIZATIONS, for example
    ``(LIBC_MATH_SKIP_ACCURATE_PASS | LIBC_MATH_SMALL_TABLES)``. Composite flags
    such as LIBC_MATH_FAST are expanded.
    """
    flags = set(re.findall(r"LIBC_MATH_[A-Z0-9_]+", value or ""))
    for composite, parts in COMPOSITE_MATH_FLAGS.items():
        if composite in flags:
            flags.update(parts)
    return flags


def parse_macros(text: str) -> Dict[str, str]:
    """Parses ``-dM -E`` output into a map of macro name to value."""
    macros = {}
    for line in text.splitlines():
        m = re.match(r"^#define\s+([A-Za-z_][A-Za-z0-9_]*)(\([^)]*\))?\s*(.*)$", line)
        if m and not m.group(2):
            macros[m.group(1)] = m.group(3).strip()
    return macros


class ManifestEntry:
    def __init__(self, function: str, macros_file: str, objects: List[str]):
        self.function = function
        self.macros_file = macros_file
        self.objects = objects


def parse_manifest(text: str) -> List[ManifestEntry]:
    """Parses a manifest of tab separated lines.

    Each line is ``<function>\\t<macros file>\\t<object>[;<object>...]``. The
    macros file holds the ``-dM -E`` output for the options the object was
    compiled with and may be empty.
    """
    entries = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) != 3:
            raise ValueError("malformed manifest line: %r" % line)
        objs = [p for p in fields[2].split(";") if p]
        if objs:
            entries.append(ManifestEntry(fields[0], fields[1], objs))
    return entries


# ===----------------------------------------------------------------------===#
# Rule evaluation.
# ===----------------------------------------------------------------------===#


def _macro_int(value: str) -> Optional[int]:
    value = value.strip().rstrip("uUlL")
    try:
        return int(value, 0)
    except ValueError:
        return None


def eval_macro_condition(cond: str, macros: Dict[str, str]) -> bool:
    """Evaluates a macro condition.

    A condition is a ``||`` separated list of alternatives. Each alternative is
    ``NAME`` (defined and not 0), ``NAME&MASK`` (defined and value & MASK is
    non-zero), optionally negated with a leading ``!``.
    """
    for alt in cond.split("||"):
        alt = alt.strip()
        negate = alt.startswith("!")
        if negate:
            alt = alt[1:].strip()
        name, _, mask = alt.partition("&")
        name = name.strip()
        result = False
        if name in macros:
            value = _macro_int(macros[name])
            if mask:
                result = value is not None and (value & int(mask.strip(), 0)) != 0
            else:
                result = value is None or value != 0
        if result != negate:
            return True
    return False


class RuleContext:
    def __init__(
        self,
        flags: Set[str],
        functions: Set[str],
        aliases: Optional[Dict[str, str]] = None,
    ):
        self.flags = flags
        self.functions = functions
        self.aliases = aliases or {}


def rule_is_active(rule: dict, ctx: RuleContext) -> bool:
    """Returns whether the configuration flags select the rule."""
    for flag in rule.get("when_flags", []):
        if flag not in ctx.flags:
            return False
    return True


def rule_macros_match(rule: dict, macros: Dict[str, str]) -> bool:
    """Returns whether an object's compile macros select the rule."""
    for cond in rule.get("when_macros", []):
        if not eval_macro_condition(cond, macros):
            return False
    return True


def _is_variant(function: str, variant: str, functions: Set[str]) -> bool:
    if variant == "f128":
        return function.endswith("f128")
    if variant == "long_double":
        return function.endswith("l") and function[:-1] in functions
    raise ValueError("unknown function variant '%s'" % variant)


def rule_applies_to(rule: dict, function: str, ctx: RuleContext) -> bool:
    patterns = rule.get("functions", [".*"])
    if not any(re.fullmatch(p, function) for p in patterns):
        return False
    if any(re.fullmatch(p, function) for p in rule.get("exclude_functions", [])):
        return False
    for variant in rule.get("exclude_variants", []):
        if _is_variant(function, variant, ctx.functions):
            return False
    return True


def _expand(pattern: str, ctx: RuleContext) -> str:
    if pattern.startswith("@"):
        if pattern[1:] not in ctx.aliases:
            raise ValueError("unknown regex alias '%s'" % pattern)
        return ctx.aliases[pattern[1:]]
    return pattern


def compile_symbol_pattern(pattern: str) -> "re.Pattern":
    """Compiles a namespace-anchored symbol pattern.

    ``pattern`` must match a trailing sequence of ``::``-separated components of
    the (prefix stripped) symbol name, e.g. ``common_constants_internal::LOG_R``
    matches ``math::common_constants_internal::LOG_R`` but neither
    ``common_constants_internal::LOG_R_DD`` nor ``xcommon_constants_internal::LOG_R``.
    """
    return re.compile(r"(?:^|::)(?:%s)$" % pattern)


def _known_exception(rule: dict, obj: ObjectInfo) -> dict:
    """Returns the known exception entry of the rule for the object, if any.

    An entry only applies when its optional ``when_macros`` conditions hold for
    the object.
    """
    entry = rule.get("known_exceptions", {}).get(obj.function)
    if entry is None:
        return {}
    for cond in entry.get("when_macros", []):
        if not eval_macro_condition(cond, obj.macros):
            return {}
    return entry


def _max_object_limit(rule: dict, function: str) -> Tuple[int, bool]:
    """Returns the named data object limit for `function` and whether it is an
    exception. Exception values are either a byte count or an object with a
    "bytes" field (and an optional "note")."""
    value = rule.get("max_named_data_object_exceptions", {}).get(function)
    if value is None:
        return rule["max_named_data_object_bytes"], False
    if isinstance(value, dict):
        value = value["bytes"]
    return value, True


def evaluate_rule(rule: dict, obj: ObjectInfo, ctx: RuleContext) -> List[Finding]:
    """Evaluates one active rule against one object it applies to."""
    rid = rule["id"]
    fn = obj.function
    findings: List[Finding] = []
    where = " in %s" % ", ".join(obj.paths)
    known_entry = _known_exception(rule, obj)
    known_patterns = known_entry.get("symbols", [])
    known = [compile_symbol_pattern(_expand(p, ctx)) for p in known_patterns]
    known_used = set()

    def violation(message: str, symbol: Optional[str] = None):
        if symbol is not None:
            for i, k in enumerate(known):
                if k.search(symbol):
                    known_used.add(i)
                    note = known_entry.get("note", "")
                    findings.append(
                        Finding(
                            rid,
                            fn,
                            STATUS_KNOWN,
                            message + (" (known exception: %s)" % note if note else ""),
                        )
                    )
                    return
        findings.append(Finding(rid, fn, STATUS_FAIL, message + where))

    for pattern in rule.get("forbid_defined", []):
        regex = compile_symbol_pattern(_expand(pattern, ctx))
        for sym in obj.defined:
            if regex.search(sym.name):
                violation(
                    "defines forbidden symbol '%s' (%d bytes)" % (sym.name, sym.size),
                    sym.name,
                )

    for pattern in rule.get("forbid_defined_regex", []):
        regex = re.compile(_expand(pattern, ctx))
        for sym in obj.defined:
            if regex.search(sym.name):
                violation(
                    "defines forbidden symbol '%s' (matches '%s')"
                    % (sym.name, pattern),
                    sym.name,
                )

    for pattern in rule.get("forbid_undefined", []):
        regex = re.compile(_expand(pattern, ctx))
        hits = sorted(u for u in obj.undefined if regex.fullmatch(u))
        for u in hits:
            violation("references forbidden helper '%s'" % u, u)

    for pattern in rule.get("require_defined", []):
        regex = compile_symbol_pattern(_expand(pattern, ctx))
        if not any(regex.search(s.name) for s in obj.defined):
            findings.append(
                Finding(
                    rid,
                    fn,
                    STATUS_FAIL,
                    "does not define a symbol matching '%s'; the rule may be stale"
                    % pattern
                    + where,
                )
            )

    if "max_named_data_object_bytes" in rule:
        limit, is_exception = _max_object_limit(rule, fn)
        largest = obj.largest_data_symbol()
        size = largest.size if largest else 0
        if size > limit:
            findings.append(
                Finding(
                    rid,
                    fn,
                    STATUS_FAIL,
                    "largest named data object '%s' is %d bytes, limit is %d bytes"
                    % (largest.name, size, limit)
                    + where,
                )
            )
        elif is_exception and size < limit:
            findings.append(
                Finding(
                    rid,
                    fn,
                    STATUS_NOTE,
                    "exception of %d bytes is larger than needed for this build "
                    "(largest named data object: %d bytes)" % (limit, size),
                )
            )

    if "max_writable_data_bytes" in rule:
        limit = rule["max_writable_data_bytes"]
        allow = [re.compile(p) for p in rule.get("allow_writable_sections", [])]
        sections = obj.writable_sections(allow)
        size = sum(s for _, s in sections)
        if size > limit:
            violation(
                "has %d bytes of writable data, limit is %d bytes: %s"
                % (size, limit, ", ".join("%s (%d)" % s for s in sections))
            )

    for i, pattern in enumerate(known_patterns):
        if i not in known_used:
            findings.append(
                Finding(
                    rid,
                    fn,
                    STATUS_NOTE,
                    "known exception '%s' no longer triggers; remove it from the rules"
                    % pattern,
                )
            )
    return findings


def evaluate(rules_doc: dict, objects: List[ObjectInfo], ctx: RuleContext):
    """Evaluates all rules. Returns (findings, summary)."""
    findings: List[Finding] = []
    summary = []
    for rule in rules_doc.get("rules", []):
        rid = rule["id"]
        if not rule_is_active(rule, ctx):
            summary.append({"rule": rid, "active": False, "checked": 0})
            continue
        exempt = set(rule.get("exceptions", []))
        checked = 0
        skipped = 0
        for obj in objects:
            if not rule_applies_to(rule, obj.function, ctx):
                continue
            if obj.function in exempt:
                continue
            if not rule_macros_match(rule, obj.macros):
                skipped += 1
                continue
            checked += 1
            findings.extend(evaluate_rule(rule, obj, ctx))
        summary.append(
            {"rule": rid, "active": True, "checked": checked, "target_skipped": skipped}
        )
    return findings, summary


# ===----------------------------------------------------------------------===#
# Linked size probes.
# ===----------------------------------------------------------------------===#

PROBE_ENTRY = "__libc_size_probe_entry"
BASELINE_PROBE = "baseline"

# C types the probe generator can pass to and receive from a function.
PROBE_TYPES = ("float", "double", "long double", "int", "long")


class Prototype:
    def __init__(self, ret: str, name: str, args: List[str]):
        self.ret = ret
        self.name = name
        self.args = args

    def declaration(self) -> str:
        return "%s %s(%s);" % (self.ret, self.name, ", ".join(self.args) or "void")


def _probe_var(c_type: str) -> str:
    return "__libc_size_probe_" + c_type.replace(" ", "_")


def parse_probe_list(
    text: str,
) -> Tuple[Dict[str, Prototype], List[Tuple[str, List[str]]]]:
    """Parses a probe list.

    Every line of the form ``<type> <name>(<types>);`` declares a function and
    adds a probe that calls it. A line ``<name>+<name>[+...]`` adds a probe that
    calls several previously declared functions, which shows how much code
    they share. ``#`` starts a comment.

    Returns the prototypes and the list of (probe name, functions).
    """
    protos: Dict[str, Prototype] = {}
    probes: List[Tuple[str, List[str]]] = []
    proto_re = re.compile(r"^(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*;$")
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = proto_re.match(line)
        if m:
            ret, name, args = m.group(1).strip(), m.group(2), m.group(3).strip()
            arg_types = [a.strip() for a in args.split(",")] if args else []
            if arg_types == ["void"]:
                arg_types = []
            for t in [ret] + arg_types:
                if t not in PROBE_TYPES:
                    raise ValueError("line %d: unsupported type '%s'" % (lineno, t))
            protos[name] = Prototype(ret, name, arg_types)
            probes.append((name, [name]))
            continue
        fns = [f.strip() for f in line.split("+")]
        if len(fns) < 2 or not all(f in protos for f in fns):
            raise ValueError("line %d: cannot parse '%s'" % (lineno, line))
        probes.append(("+".join(fns), fns))
    return protos, probes


def generate_probe_source(functions: List[str], protos: Dict[str, Prototype]) -> str:
    """Generates a C program whose entry point calls `functions`.

    The arguments and results go through volatile globals so that the calls
    cannot be optimized away. The same globals and the errno stub are part of
    every probe, including the baseline, so they cancel out in the deltas.
    """
    out = ["/* Generated by libc_size_check.py. */"]
    for fn in functions:
        out.append(protos[fn].declaration())
    # 'retain' keeps the globals and the errno stub alive with --gc-sections
    # even when they are not referenced, so that they cancel out exactly.
    keep = "__attribute__((used, retain))"
    for t in PROBE_TYPES:
        out.append("%s volatile %s %s;" % (keep, t, _probe_var(t)))
    # Needed with LIBC_ERRNO_MODE_EXTERNAL. Weak so that a definition in libc
    # takes precedence.
    out.append("%s __attribute__((weak)) int *__llvm_libc_errno(void) {" % keep)
    out.append("  static int e;")
    out.append("  return &e;")
    out.append("}")
    out.append("void %s(void) {" % PROBE_ENTRY)
    for fn in functions:
        p = protos[fn]
        call = "%s(%s)" % (fn, ", ".join(_probe_var(a) for a in p.args))
        out.append("  %s = %s;" % (_probe_var(p.ret), call))
    out.append("}")
    return "\n".join(out) + "\n"


class Section:
    def __init__(self, name: str, sec_type: str, size: int, flags: str):
        self.name = name
        self.type = sec_type
        self.size = size
        self.flags = flags

    def is_alloc(self) -> bool:
        return "A" in self.flags

    def is_flash(self) -> bool:
        return self.is_alloc() and self.type != "NOBITS"

    def is_ram(self) -> bool:
        return self.is_alloc() and "W" in self.flags


def parse_readelf_sections(text: str) -> List[Section]:
    """Parses the section headers printed by ``llvm-readelf -S -W``."""
    sections = []
    for line in text.splitlines():
        m = re.match(r"^\s*\[\s*\d+\]\s(.*)$", line)
        if not m:
            continue
        fields = m.group(1).split()
        # Name Type Address Off Size ES [Flg] Lk Inf Al
        if len(fields) == 10:
            name, sec_type, size, flags = fields[0], fields[1], fields[4], fields[6]
        elif len(fields) == 9:
            name, sec_type, size, flags = fields[0], fields[1], fields[4], ""
        else:
            continue  # The NULL section has no name.
        sections.append(Section(name, sec_type, int(size, 16), flags))
    return sections


_MAP_LINE_RE = re.compile(
    r"^\s*([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+(\d+) ( *)(\S.*)$"
)


def parse_lld_map(text: str) -> List[Tuple[str, str, int]]:
    """Parses an lld map file into (output section, input section, size).

    Only input section lines are returned; symbol lines are skipped.
    """
    entries = []
    output = None
    for line in text.splitlines():
        m = _MAP_LINE_RE.match(line)
        if not m:
            continue
        depth = len(m.group(5)) // 8
        if depth == 0:
            output = m.group(6).strip()
        elif depth == 1 and output is not None:
            entries.append((output, m.group(6).strip(), int(m.group(3), 16)))
    return entries


def probe_metrics(
    sections: List[Section],
    map_entries: List[Tuple[str, str, int]],
    data_symbols: List[Symbol],
    archives: List[str],
) -> dict:
    """Computes the size metrics of a linked probe.

    flash:      allocated sections with file contents (code and read-only data).
    ram:        allocated writable sections (.data and .bss).
    libc_flash: the part of `flash` that comes from the libc archives.
    named_data: total size of the named data objects (tables), which does not
                depend on the compiler.

    flash and ram are sums of input sections, so that alignment padding, which
    depends on the section order, does not add noise.
    """
    flash_sections = {s.name for s in sections if s.is_flash()}
    ram_sections = {s.name for s in sections if s.is_ram()}
    flash = ram = libc_flash = 0
    members = set()
    for output, inp, size in map_entries:
        if output in flash_sections:
            flash += size
        if output in ram_sections:
            ram += size
        for archive in archives:
            if inp.startswith(archive + "("):
                member = inp[len(archive) + 1 :].split(")", 1)[0]
                members.add("%s(%s)" % (os.path.basename(archive), member))
                if output in flash_sections:
                    libc_flash += size
    return {
        "flash": flash,
        "ram": ram,
        "libc_flash": libc_flash,
        "named_data": sum(s.size for s in data_symbols),
        "members": sorted(members),
    }


def probe_delta(metrics: dict, baseline: dict) -> dict:
    delta = {}
    for key in ("flash", "ram", "libc_flash", "named_data"):
        delta[key] = metrics[key] - baseline[key]
    delta["other_flash"] = delta["flash"] - delta["libc_flash"]
    delta["members"] = [m for m in metrics["members"] if m not in baseline["members"]]
    return delta


_UNDEFINED_RE = re.compile(r"undefined symbol: (\S+)")


class ProbeBuilder:
    """Compiles and links probe programs and measures them."""

    def __init__(self, args):
        self.args = args
        self.archives = [a for a in args.archive if a]

    def _run(self, cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    def build(self, name: str, source: str) -> dict:
        """Builds one probe. Returns its metrics or a 'skipped' reason."""
        base = os.path.join(self.args.work_dir, name.replace("+", "_"))
        write_file(base + ".c", source)
        compile_cmd = [self.args.cc] + self.args.cflag
        compile_cmd += ["-Os", "-ffunction-sections", "-fdata-sections"]
        compile_cmd += ["-ffreestanding", "-fno-builtin"]
        compile_cmd += ["-fno-asynchronous-unwind-tables", "-fno-unwind-tables"]
        compile_cmd += ["-c", base + ".c", "-o", base + ".o"]
        result = self._run(compile_cmd)
        if result.returncode != 0:
            raise RuntimeError("compiling %s failed:\n%s" % (name, result.stdout))
        link_cmd = [self.args.cc] + self.args.cflag + self.args.ldflag
        link_cmd += ["-nostdlib", "-static", "-Wl,--gc-sections", "-Wl,-z,norelro"]
        link_cmd += ["-Wl,-e," + PROBE_ENTRY, "-Wl,-Map=" + base + ".map"]
        link_cmd += [base + ".o", "-Wl,--start-group"] + self.archives
        link_cmd += ["-Wl,--end-group"]
        if self.args.builtins:
            link_cmd.append(self.args.builtins)
        link_cmd += ["-o", base + ".elf"]
        result = self._run(link_cmd)
        if result.returncode != 0:
            undefined = sorted(set(_UNDEFINED_RE.findall(result.stdout)))
            if undefined and not self.args.builtins:
                return {
                    "skipped": "unresolved symbols and no compiler runtime builtins "
                    "library: %s"
                    % ", ".join(undefined[:4] + (["..."] if len(undefined) > 4 else []))
                }
            raise RuntimeError("linking %s failed:\n%s" % (name, result.stdout))
        sections = parse_readelf_sections(
            run_tool([self.args.readelf, "-S", "-W", base + ".elf"])
        )
        defined, _ = parse_nm(
            run_tool([self.args.nm, "-S", "-C", base + ".elf"]),
            re.compile(DEFAULT_SYMBOL_PREFIX_REGEX),
        )
        data = [s for s in defined if s.is_data()]
        map_entries = parse_lld_map(read_file(base + ".map"))
        return probe_metrics(sections, map_entries, data, self.archives)


def format_probe_text(results: List[Tuple[str, dict]]) -> str:
    lines = [
        "%-24s %8s %8s %8s %8s %6s  %s"
        % ("probe", "flash", "libc", "other", "tables", "ram", "libc objects")
    ]
    for name, r in results:
        if "skipped" in r:
            lines.append("%-24s skipped: %s" % (name, r["skipped"]))
            continue
        lines.append(
            "%-24s %8d %8d %8d %8d %6d  %s"
            % (
                name,
                r["flash"],
                r["libc_flash"],
                r["other_flash"],
                r["named_data"],
                r["ram"],
                " ".join(m.split("(", 1)[1].rstrip(")") for m in r["members"]),
            )
        )
    return "\n".join(lines)


def format_probe_markdown(results: List[Tuple[str, dict]], title: str) -> str:
    out = ["# %s" % title, ""]
    out.append(
        "Sizes in bytes relative to an empty program. `libc` is the part of "
        "`flash` that comes from the libc archives, `other` is mostly compiler "
        "runtime builtins, `tables` is the size of the named data objects."
    )
    out += ["", "| probe | flash | libc | other | tables | ram | status |"]
    out.append("|---|---:|---:|---:|---:|---:|---|")
    for name, r in results:
        if "skipped" in r:
            out.append("| %s | | | | | | skipped: %s |" % (name, r["skipped"]))
        else:
            out.append(
                "| %s | %d | %d | %d | %d | %d | ok |"
                % (
                    name,
                    r["flash"],
                    r["libc_flash"],
                    r["other_flash"],
                    r["named_data"],
                    r["ram"],
                )
            )
    return "\n".join(out) + "\n"


def cmd_probes(args) -> int:
    protos, probes = parse_probe_list(read_file(args.probes))
    available = None
    if args.manifest:
        available = {e.function for e in parse_manifest(read_file(args.manifest))}
    os.makedirs(args.work_dir, exist_ok=True)
    builder = ProbeBuilder(args)
    baseline = builder.build(BASELINE_PROBE, generate_probe_source([], protos))
    if "skipped" in baseline:
        raise RuntimeError(
            "the baseline probe failed to link: %s" % baseline["skipped"]
        )

    def build(probe):
        name, fns = probe
        missing = [f for f in fns if available is not None and f not in available]
        if missing:
            return name, {"skipped": "not in the build: %s" % ", ".join(missing)}
        result = builder.build(name, generate_probe_source(fns, protos))
        return name, result if "skipped" in result else probe_delta(result, baseline)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(build, probes))

    print("libc size probes (report only): sizes in bytes relative to an empty program")
    if not args.builtins:
        print(
            "note: no compiler runtime builtins library; probes that need one are "
            "skipped (set LIBC_TEST_BUILTINS_LIBRARY or LIBC_TEST_BUILTINS_TARGET)"
        )
    print(format_probe_text(results))
    if args.json:
        write_file(
            args.json,
            json.dumps(
                {"baseline": baseline, "probes": dict(results)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    if args.markdown:
        write_file(args.markdown, format_probe_markdown(results, "libc size probes"))
    skipped = sum(1 for _, r in results if "skipped" in r)
    print(
        "libc size probes: %d probes linked, %d skipped."
        % (len(results) - skipped, skipped)
    )
    return 0


# ===----------------------------------------------------------------------===#
# Driver.
# ===----------------------------------------------------------------------===#


def run_tool(cmd: List[str]) -> str:
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            "command failed (%d): %s\n%s"
            % (result.returncode, " ".join(cmd), result.stderr)
        )
    return result.stdout


def load_object(
    entry: ManifestEntry,
    nm: str,
    size: str,
    prefix_re: "re.Pattern",
    macros: Dict[str, str],
) -> ObjectInfo:
    defined: List[Symbol] = []
    undefined: Set[str] = set()
    sections: Dict[str, int] = {}
    for path in entry.objects:
        d, u = parse_nm(run_tool([nm, "-S", "-C", path]), prefix_re)
        defined.extend(d)
        undefined.update(u)
        for name, sz in parse_size_sysv(run_tool([size, "-A", path])).items():
            sections[name] = sections.get(name, 0) + sz
    # A symbol defined by one of the objects is not an external reference.
    undefined -= {s.raw_name for s in defined}
    return ObjectInfo(
        entry.function, entry.objects, defined, undefined, sections, macros
    )


def load_objects(
    manifest: List[ManifestEntry], nm: str, size: str, prefix_re, jobs: int
) -> List[ObjectInfo]:
    macro_cache: Dict[str, Dict[str, str]] = {}
    for entry in manifest:
        if entry.macros_file and entry.macros_file not in macro_cache:
            macro_cache[entry.macros_file] = parse_macros(read_file(entry.macros_file))
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                load_object,
                entry,
                nm,
                size,
                prefix_re,
                macro_cache.get(entry.macros_file, {}),
            )
            for entry in manifest
        ]
        return [f.result() for f in futures]


def _count(findings: List[Finding], rule_id: str, status: str) -> int:
    return sum(1 for f in findings if f.rule_id == rule_id and f.status == status)


def format_text(findings: List[Finding], summary: List[dict], verbose=False) -> str:
    """Formats the findings for the terminal.

    Failures are always listed; known exceptions and notes only when `verbose`.
    """
    lines = []
    for f in findings:
        if f.status == STATUS_FAIL or verbose:
            lines.append(
                "%s: [%s] %s: %s" % (f.status.upper(), f.rule_id, f.function, f.message)
            )
    active = [s for s in summary if s["active"]]
    inactive = [s["rule"] for s in summary if not s["active"]]
    for s in active:
        rid = s["rule"]
        lines.append(
            "rule %-36s checked %4d objects: %d failures, %d known exceptions, "
            "%d notes%s"
            % (
                rid,
                s["checked"],
                _count(findings, rid, STATUS_FAIL),
                _count(findings, rid, STATUS_KNOWN),
                _count(findings, rid, STATUS_NOTE),
                (
                    " (%d skipped by target macros)" % s["target_skipped"]
                    if s.get("target_skipped")
                    else ""
                ),
            )
        )
    if inactive:
        lines.append("inactive rules (flags not set): %s" % ", ".join(inactive))
    return "\n".join(lines)


def format_markdown(
    findings: List[Finding], summary: List[dict], objects: List[ObjectInfo]
) -> str:
    out = ["# libc size check report", ""]
    out.append("| rule | active | checked | failures | known | notes |")
    out.append("|---|---|---:|---:|---:|---:|")
    for s in summary:
        rid = s["rule"]
        out.append(
            "| %s | %s | %d | %d | %d | %d |"
            % (
                rid,
                "yes" if s["active"] else "no",
                s["checked"],
                _count(findings, rid, STATUS_FAIL),
                _count(findings, rid, STATUS_KNOWN),
                _count(findings, rid, STATUS_NOTE),
            )
        )
    if findings:
        out += ["", "## Findings", "", "| status | rule | function | message |"]
        out.append("|---|---|---|---|")
        for f in findings:
            out.append(
                "| %s | %s | %s | %s |"
                % (f.status, f.rule_id, f.function, f.message.replace("|", "\\|"))
            )
    out += ["", "## Objects", "", "| function | bytes | largest data object | bytes |"]
    out.append("|---|---:|---|---:|")
    for obj in sorted(objects, key=lambda o: o.function):
        largest = obj.largest_data_symbol()
        out.append(
            "| %s | %d | %s | %d |"
            % (
                obj.function,
                obj.total_bytes(),
                largest.name if largest else "",
                largest.size if largest else 0,
            )
        )
    return "\n".join(out) + "\n"


def objects_report_json(objects: List[ObjectInfo]) -> dict:
    result = {}
    for obj in objects:
        largest = obj.largest_data_symbol()
        result[obj.function] = {
            "bytes": obj.total_bytes(),
            "writable_bytes": obj.writable_bytes(),
            "named_data_bytes": sum(s.size for s in obj.data_symbols()),
            "largest_data_object": largest.name if largest else None,
            "largest_data_object_bytes": largest.size if largest else 0,
        }
    return result


def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def cmd_objects(args) -> int:
    rules_doc = json.loads(read_file(args.rules))
    prefix_re = re.compile(
        rules_doc.get("symbol_prefix_regex", DEFAULT_SYMBOL_PREFIX_REGEX)
    )
    manifest = parse_manifest(read_file(args.manifest))
    flags = parse_math_flags(args.math_optimizations)
    ctx = RuleContext(
        flags, {e.function for e in manifest}, rules_doc.get("regex_aliases", {})
    )
    objects = load_objects(manifest, args.nm, args.size, prefix_re, args.jobs)
    findings, summary = evaluate(rules_doc, objects, ctx)

    print(
        "libc size check: %d objects, LIBC_MATH flags: %s"
        % (len(objects), " ".join(sorted(flags)) or "(none)")
    )
    print(format_text(findings, summary, args.verbose))
    if args.json:
        write_file(
            args.json,
            json.dumps(
                {
                    "math_flags": sorted(flags),
                    "summary": summary,
                    "findings": [f.to_json() for f in findings],
                    "objects": objects_report_json(objects),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    if args.markdown:
        write_file(args.markdown, format_markdown(findings, summary, objects))

    failures = [f for f in findings if f.status == STATUS_FAIL]
    if failures:
        print(
            "libc size check: %d rule violation(s)%s. See %s for the rules."
            % (
                len(failures),
                " (report only, not failing)" if args.report_only else "",
                args.rules,
            )
        )
        return 0 if args.report_only else 1
    print("libc size check: all rules passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p = sub.add_parser("objects", help="check entrypoint objects against rules")
    p.add_argument("--manifest", required=True, help="see parse_manifest()")
    p.add_argument("--rules", required=True, help="rules JSON file")
    p.add_argument(
        "--math-optimizations",
        default="",
        help="value of LIBC_CONF_MATH_OPTIMIZATIONS",
    )
    p.add_argument("--nm", default="llvm-nm", help="path to llvm-nm")
    p.add_argument("--size", default="llvm-size", help="path to llvm-size")
    p.add_argument("--json", help="write a JSON report to this file")
    p.add_argument("--markdown", help="write a markdown report to this file")
    p.add_argument("--report-only", action="store_true", help="never fail")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also list known exceptions and notes",
    )
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    p.set_defaults(func=cmd_objects)

    p = sub.add_parser("probes", help="link and measure minimal programs")
    p.add_argument("--probes", required=True, help="probe list, see parse_probe_list()")
    p.add_argument("--work-dir", required=True, help="directory for the probes")
    p.add_argument(
        "--manifest", help="object manifest; probes of other functions are skipped"
    )
    p.add_argument("--cc", required=True, help="C compiler driver")
    p.add_argument("--cflag", action="append", default=[], help="compile and link flag")
    p.add_argument("--ldflag", action="append", default=[], help="link flag")
    p.add_argument(
        "--archive", action="append", default=[], help="libc archive to link"
    )
    p.add_argument("--builtins", default="", help="compiler runtime builtins library")
    p.add_argument("--nm", default="llvm-nm", help="path to llvm-nm")
    p.add_argument("--readelf", default="llvm-readelf", help="path to llvm-readelf")
    p.add_argument("--json", help="write a JSON report to this file")
    p.add_argument("--markdown", help="write a markdown report to this file")
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    p.set_defaults(func=cmd_probes)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
