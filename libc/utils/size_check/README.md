# libc binary size checks

This directory contains a checker for the binary size of LLVM libc entrypoints.
It catches size regressions that the functional tests cannot see, for example a
large lookup table or the multi-precision accurate pass creeping back into a
configuration that asked for `LIBC_MATH_SMALL_TABLES` or
`LIBC_MATH_SKIP_ACCURATE_PASS`.

The checks never execute target code. They only inspect objects with
`llvm-nm` and `llvm-size`, so they work the same way for host, cross and
baremetal builds.

## The checker

`libc_size_check.py objects` reads a manifest of entrypoint objects, runs
`llvm-nm -S -C` and `llvm-size -A` on each object and evaluates the rules of a
rules file against them:

```
libc_size_check.py objects --manifest <file> --rules rules/math.json \
    --math-optimizations "<value of LIBC_CONF_MATH_OPTIMIZATIONS>" \
    --nm <llvm-nm> --size <llvm-size> [--json <report>] [--markdown <report>] \
    [--report-only] [--verbose]
```

Each manifest line has three tab separated fields: the entrypoint name, a file
with the `-dM -E` output of the compiler for the options the object was built
with (may be empty), and a `;` separated list of object files.

The checker prints every rule violation and a summary line per rule, and exits
with a non-zero status if a rule is violated (unless `--report-only` is given).
Known exceptions and advisory notes (for example an exception that is no longer
needed) are printed with `--verbose` and are always part of the JSON and
markdown reports.

The unit tests run on canned tool output and need no toolchain:

```
python3 libc/utils/size_check/libc_size_check_test.py
```

## Rules files

A rules file is a JSON object:

```json
{
  "version": 1,
  "symbol_prefix_regex": "^__llvm_libc[0-9A-Za-z_]*::",
  "regex_aliases": {"soft_double_helpers": "__aeabi_dadd|..."},
  "rules": [ ... ]
}
```

Symbol names are demangled, and the versioned libc namespace prefix matched by
`symbol_prefix_regex` is removed, so rules can name
`common_constants_internal::LOG_R` instead of
`__llvm_libc_24_0_0_git::common_constants_internal::LOG_R`. A pattern of the
form `@name` refers to an entry of `regex_aliases`.

Each rule has an `id`, an optional `comment`, and these optional fields:

| field | meaning |
|---|---|
| `when_flags` | `LIBC_MATH_*` flags that must all be set in `LIBC_CONF_MATH_OPTIMIZATIONS` (composite flags like `LIBC_MATH_FAST` are expanded). Otherwise the rule is inactive. |
| `when_macros` | Conditions on the predefined macros of each object's compile options, all of which must hold. A condition is a `\|\|` separated list of `NAME` (defined and not 0) or `NAME&MASK` (value has a bit of MASK set), each optionally negated with `!`. Example: `"!__ARM_FP&0x8"` (no hardware double precision). |
| `functions` | Regular expressions; the rule applies to the entrypoints fully matching one of them. Defaults to all entrypoints. |
| `exclude_functions` | Regular expressions of entrypoints to skip. |
| `exclude_variants` | `f128` skips `*f128` entrypoints, `long_double` skips `<name>l` when `<name>` is also an entrypoint. |
| `exceptions` | Entrypoints the rule does not apply to. |
| `forbid_defined` | Symbol patterns that must not be defined. A pattern matches trailing `::` separated components of the symbol name: `common_constants_internal::LOG_R` matches `math::common_constants_internal::LOG_R` but not `common_constants_internal::LOG_R_DD`. |
| `forbid_defined_regex` | Regular expressions searched in the defined symbol names. |
| `forbid_undefined` | Regular expressions that must not fully match an undefined symbol, e.g. soft-float helpers. |
| `require_defined` | Symbol patterns that must be defined. They detect rules that silently stopped matching because a table was renamed. |
| `max_named_data_object_bytes` | Limit for the largest named data object (table) of an object. |
| `max_named_data_object_exceptions` | Per-entrypoint limits, either a number of bytes or `{"bytes": N, "note": "..."}`. |
| `max_writable_data_bytes` | Limit for the size of the `.data`, `.bss`, `.tdata` and `.tbss` sections. |
| `allow_writable_sections` | Regular expressions of writable sections not counted against the limit. |
| `known_exceptions` | `{"<entrypoint>": {"symbols": [...], "note": "...", "when_macros": [...]}}`. Violations caused by the listed symbols are reported as known exceptions instead of failures. Use them for regressions that are already being worked on; the checker reports exceptions that no longer trigger so they can be removed. |
