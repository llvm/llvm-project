#!/usr/bin/env python3
#
# ===- libc_size_check_test.py - Tests for libc_size_check.py --*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#
"""Unit tests for libc_size_check.py on canned tool output.

No toolchain is needed: run with `python3 libc_size_check_test.py`.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import libc_size_check as lsc

PREFIX_RE = re.compile(lsc.DEFAULT_SYMBOL_PREFIX_REGEX)

# `llvm-nm -S -C` output of a logf object for an ARM soft-float target.
NM_LOGF_ARM = """\
00000000 00000468 T logf
00000000 00000080 V __llvm_libc_24_0_0_git::common_constants_internal::LOG_R
00000000 00000004 r __llvm_libc_24_0_0_git::math::logf()::SMALL
         U __aeabi_dadd
         U __aeabi_f2d
         U __aeabi_fmul
         w __llvm_libc_24_0_0_git::weak_hook()
00000000 T __llvm_libc_24_0_0_git::logf(float)
"""

# `llvm-nm -S -C` output of a pow object for x86-64 in the default config.
NM_POW_X86 = """\
0000000000000000 0000000000000a2c T pow
0000000000000000 0000000000001000 V __llvm_libc_24_0_0_git::pow_internal::LOG2_RD_FRAC256
0000000000000000 0000000000000120 W __llvm_libc_24_0_0_git::fputil::DyadicFloat<128ul>::operator*(__llvm_libc_24_0_0_git::fputil::DyadicFloat<128ul> const&) const
0000000000000000 0000000000000004 d __llvm_libc_24_0_0_git::fputil::generic::quick_get_round()::x
                 U __llvm_libc_errno
"""

# `llvm-size -A` output.
SIZE_POW_X86 = """\
pow.cpp.o  :
section                                                         size   addr
.text                                                              0      0
.text.pow                                                       2604      0
.rodata._ZN22__llvm_libc_24_0_0_git12pow_internal15LOG2_RD_FRAC256E  4096      0
.rodata.cst8                                                      48      0
.rodata.cst8                                                      16      0
.data._ZZN22__llvm_libc_24_0_0_git6fputil7generic15quick_get_roundEvE1x     4      0
.bss                                                               0      0
.comment                                                         102      0
.note.GNU-stack                                                    0      0
.llvm_addrsig                                                      4      0
Total                                                           6874
"""

MACROS_ARMV7EM = """\
#define __arm__ 1
#define __ARM_ARCH 7
#define __ARM_FP 0x4
#define __ARM_ARCH_PROFILE 'M'
#define __SIZEOF_DOUBLE__ 8
#define __GNUC_PREREQ(a, b) 1
"""

MACROS_ARMV81M = """\
#define __arm__ 1
#define __ARM_FP 0xe
#define __ARM_FEATURE_FMA 1
"""


def make_obj(function, nm_text, size_text="", macros=None):
    defined, undefined = lsc.parse_nm(nm_text, PREFIX_RE)
    return lsc.ObjectInfo(
        function,
        ["%s.cpp.o" % function],
        defined,
        undefined - {s.raw_name for s in defined},
        lsc.parse_size_sysv(size_text),
        macros,
    )


def statuses(findings, status=lsc.STATUS_FAIL):
    return [(f.rule_id, f.function) for f in findings if f.status == status]


class ParseTest(unittest.TestCase):
    def test_parse_nm(self):
        defined, undefined = lsc.parse_nm(NM_LOGF_ARM, PREFIX_RE)
        names = {s.name: s for s in defined}
        self.assertIn("common_constants_internal::LOG_R", names)
        self.assertEqual(names["common_constants_internal::LOG_R"].size, 128)
        self.assertEqual(names["common_constants_internal::LOG_R"].type, "V")
        self.assertEqual(names["math::logf()::SMALL"].size, 4)
        # Symbols without a size are still recorded.
        self.assertEqual(names["logf(float)"].size, 0)
        self.assertEqual(
            undefined,
            {
                "__aeabi_dadd",
                "__aeabi_f2d",
                "__aeabi_fmul",
                "__llvm_libc_24_0_0_git::weak_hook()",
            },
        )

    def test_parse_nm_names_with_spaces(self):
        defined, _ = lsc.parse_nm(NM_POW_X86, PREFIX_RE)
        names = [s.name for s in defined]
        self.assertIn(
            "fputil::DyadicFloat<128ul>::operator*(__llvm_libc_24_0_0_git::fputil::"
            "DyadicFloat<128ul> const&) const",
            names,
        )

    def test_parse_nm_ignores_file_headers(self):
        defined, undefined = lsc.parse_nm(
            "\nfoo.o:\n00000000 00000004 R x\n", PREFIX_RE
        )
        self.assertEqual([s.name for s in defined], ["x"])
        self.assertEqual(undefined, set())

    def test_parse_size(self):
        sections = lsc.parse_size_sysv(SIZE_POW_X86)
        self.assertEqual(sections[".text.pow"], 2604)
        self.assertEqual(sections[".rodata.cst8"], 64)  # Summed.
        self.assertNotIn("Total", sections)
        obj = make_obj("pow", NM_POW_X86, SIZE_POW_X86)
        self.assertEqual(obj.writable_bytes(), 4)
        self.assertEqual(obj.total_bytes(), 2604 + 4096 + 64 + 4)

    def test_largest_data_symbol(self):
        obj = make_obj("pow", NM_POW_X86)
        largest = obj.largest_data_symbol()
        self.assertEqual(largest.name, "pow_internal::LOG2_RD_FRAC256")
        self.assertEqual(largest.size, 4096)
        # Weak functions (W) are not data.
        self.assertEqual(len(obj.data_symbols()), 2)

    def test_parse_math_flags(self):
        self.assertEqual(lsc.parse_math_flags(""), set())
        self.assertEqual(
            lsc.parse_math_flags(
                "(LIBC_MATH_SKIP_ACCURATE_PASS | LIBC_MATH_SMALL_TABLES)"
            ),
            {"LIBC_MATH_SKIP_ACCURATE_PASS", "LIBC_MATH_SMALL_TABLES"},
        )
        fast = lsc.parse_math_flags("LIBC_MATH_FAST")
        self.assertIn("LIBC_MATH_SMALL_TABLES", fast)
        self.assertIn("LIBC_MATH_ASSUME_ROUND_NEAREST_ONLY", fast)
        self.assertNotIn("LIBC_MATH_INTERMEDIATE_COMP_IN_FLOAT", fast)

    def test_parse_macros(self):
        macros = lsc.parse_macros(MACROS_ARMV7EM)
        self.assertEqual(macros["__ARM_FP"], "0x4")
        self.assertEqual(macros["__ARM_ARCH_PROFILE"], "'M'")
        self.assertNotIn("__GNUC_PREREQ", macros)  # Function-like macro.

    def test_parse_manifest(self):
        entries = lsc.parse_manifest(
            "# comment\nlogf\t/m/a.txt\t/o/logf.o\n"
            "powf\t\t/o/powf1.o;/o/powf2.o\nempty\t/m/a.txt\t\n"
        )
        self.assertEqual([e.function for e in entries], ["logf", "powf"])
        self.assertEqual(entries[0].macros_file, "/m/a.txt")
        self.assertEqual(entries[1].objects, ["/o/powf1.o", "/o/powf2.o"])
        with self.assertRaises(ValueError):
            lsc.parse_manifest("logf /o/logf.o\n")


class MacroConditionTest(unittest.TestCase):
    def setUp(self):
        self.v7em = lsc.parse_macros(MACROS_ARMV7EM)
        self.v81m = lsc.parse_macros(MACROS_ARMV81M)

    def test_defined(self):
        self.assertTrue(lsc.eval_macro_condition("__arm__", self.v7em))
        self.assertFalse(lsc.eval_macro_condition("__x86_64__", self.v7em))
        self.assertTrue(lsc.eval_macro_condition("!__x86_64__", self.v7em))
        # Non-integer values count as defined.
        self.assertTrue(lsc.eval_macro_condition("__ARM_ARCH_PROFILE", self.v7em))

    def test_zero_is_false(self):
        self.assertFalse(lsc.eval_macro_condition("ZERO", {"ZERO": "0"}))

    def test_mask(self):
        # __ARM_FP bit 3 (0x8) means hardware double precision.
        self.assertFalse(lsc.eval_macro_condition("__ARM_FP&0x8", self.v7em))
        self.assertTrue(lsc.eval_macro_condition("__ARM_FP&0x8", self.v81m))
        self.assertTrue(lsc.eval_macro_condition("!__ARM_FP&0x8", self.v7em))
        self.assertTrue(lsc.eval_macro_condition("!__ARM_FP&0x8", {}))

    def test_or(self):
        cond = "__FMA__ || __ARM_FEATURE_FMA"
        self.assertFalse(lsc.eval_macro_condition(cond, self.v7em))
        self.assertTrue(lsc.eval_macro_condition(cond, self.v81m))
        self.assertTrue(lsc.eval_macro_condition(cond, {"__FMA__": "1"}))


class SymbolPatternTest(unittest.TestCase):
    def test_namespace_anchored(self):
        p = lsc.compile_symbol_pattern("common_constants_internal::LOG_R")
        self.assertTrue(p.search("common_constants_internal::LOG_R"))
        self.assertTrue(p.search("math::common_constants_internal::LOG_R"))
        self.assertFalse(p.search("common_constants_internal::LOG_R_DD"))
        self.assertFalse(p.search("xcommon_constants_internal::LOG_R"))

    def test_alternation(self):
        p = lsc.compile_symbol_pattern("EXP2_MID1|EXP2_MID2")
        self.assertTrue(p.search("EXP2_MID1"))
        self.assertTrue(p.search("math::EXP2_MID2"))
        self.assertFalse(p.search("EXP2_MID"))


class RuleTest(unittest.TestCase):
    ALIASES = {"soft_double": "__aeabi_(dadd|dmul|f2d|d2f)"}

    def evaluate(self, rules, objects, flags=()):
        ctx = lsc.RuleContext(set(flags), {o.function for o in objects}, self.ALIASES)
        return lsc.evaluate({"rules": rules}, objects, ctx)

    def test_when_flags(self):
        rule = {"id": "r", "when_flags": ["LIBC_MATH_SMALL_TABLES"]}
        rule["forbid_defined"] = ["common_constants_internal::LOG_R"]
        obj = make_obj("logf", NM_LOGF_ARM)
        findings, summary = self.evaluate([rule], [obj])
        self.assertEqual(findings, [])
        self.assertFalse(summary[0]["active"])
        findings, summary = self.evaluate([rule], [obj], ["LIBC_MATH_SMALL_TABLES"])
        self.assertEqual(statuses(findings), [("r", "logf")])
        self.assertIn("common_constants_internal::LOG_R", findings[0].message)
        self.assertIn("128 bytes", findings[0].message)
        self.assertIn("logf.cpp.o", findings[0].message)

    def test_when_macros_per_object(self):
        rule = {
            "id": "r",
            "when_macros": ["!__ARM_FP&0x8"],
            "forbid_undefined": ["@soft_double"],
        }
        v7em = make_obj("logf", NM_LOGF_ARM, macros=lsc.parse_macros(MACROS_ARMV7EM))
        findings, summary = self.evaluate([rule], [v7em])
        self.assertEqual(
            sorted(f.message.split("'")[1] for f in findings),
            ["__aeabi_dadd", "__aeabi_f2d"],
        )
        v81m = make_obj("logf", NM_LOGF_ARM, macros=lsc.parse_macros(MACROS_ARMV81M))
        findings, summary = self.evaluate([rule], [v81m])
        self.assertEqual(findings, [])
        self.assertEqual(summary[0]["checked"], 0)
        self.assertEqual(summary[0]["target_skipped"], 1)

    def test_functions_and_exceptions(self):
        rule = {
            "id": "r",
            "functions": ["log.*"],
            "exceptions": ["log2f"],
            "forbid_defined": ["common_constants_internal::LOG_R"],
        }
        objs = [
            make_obj("logf", NM_LOGF_ARM),
            make_obj("log2f", NM_LOGF_ARM),
            make_obj("expf", NM_LOGF_ARM),
        ]
        findings, summary = self.evaluate([rule], objs)
        self.assertEqual(statuses(findings), [("r", "logf")])
        self.assertEqual(summary[0]["checked"], 1)

    def test_exclude_variants(self):
        rule = {
            "id": "r",
            "exclude_variants": ["f128", "long_double"],
            "forbid_defined_regex": ["DyadicFloat<128"],
        }
        objs = [
            make_obj("pow", NM_POW_X86),
            make_obj("powl", NM_POW_X86),
            make_obj("powf128", NM_POW_X86),
            # "ceil" ends in 'l' but "cei" is not a function.
            make_obj("ceil", NM_POW_X86),
        ]
        findings, _ = self.evaluate([rule], objs)
        self.assertEqual(statuses(findings), [("r", "pow"), ("r", "ceil")])

    def test_known_exceptions(self):
        rule = {
            "id": "r",
            "forbid_defined": ["common_constants_internal::LOG_R", "UNUSED_TABLE"],
            "known_exceptions": {
                "logf": {
                    "symbols": ["common_constants_internal::LOG_R", "UNUSED_TABLE"],
                    "note": "being fixed",
                }
            },
        }
        findings, _ = self.evaluate([rule], [make_obj("logf", NM_LOGF_ARM)])
        self.assertEqual(statuses(findings), [])
        self.assertEqual(statuses(findings, lsc.STATUS_KNOWN), [("r", "logf")])
        self.assertIn("being fixed", findings[0].message)
        notes = [f.message for f in findings if f.status == lsc.STATUS_NOTE]
        self.assertEqual(len(notes), 1)
        self.assertIn("UNUSED_TABLE", notes[0])

    def test_require_defined(self):
        rule = {"id": "r", "require_defined": ["log2f_float_internal::R_16"]}
        findings, _ = self.evaluate([rule], [make_obj("logf", NM_LOGF_ARM)])
        self.assertEqual(statuses(findings), [("r", "logf")])
        self.assertIn("stale", findings[0].message)

    def test_max_named_data_object(self):
        rule = {
            "id": "r",
            "max_named_data_object_bytes": 512,
            "max_named_data_object_exceptions": {
                "pow": 4096,
                "logf": {"bytes": 1024, "note": "being fixed"},
            },
        }
        objs = [
            make_obj("pow", NM_POW_X86),
            make_obj("logf", NM_LOGF_ARM),
            make_obj("powx", NM_POW_X86),
        ]
        findings, _ = self.evaluate([rule], objs)
        self.assertEqual(statuses(findings), [("r", "powx")])
        fails = [f for f in findings if f.status == lsc.STATUS_FAIL]
        self.assertIn("4096 bytes, limit is 512", fails[0].message)
        # The logf exception is larger than needed for this build.
        notes = [f for f in findings if f.status == lsc.STATUS_NOTE]
        self.assertEqual([(f.function) for f in notes], ["logf"])
        self.assertIn("exception of 1024 bytes is larger than needed", notes[0].message)

    def test_max_writable_data(self):
        rule = {"id": "r", "max_writable_data_bytes": 0}
        findings, _ = self.evaluate([rule], [make_obj("pow", NM_POW_X86, SIZE_POW_X86)])
        self.assertEqual(statuses(findings), [("r", "pow")])
        self.assertIn("4 bytes of writable data", findings[0].message)
        rule["allow_writable_sections"] = ["quick_get_round"]
        findings, _ = self.evaluate([rule], [make_obj("pow", NM_POW_X86, SIZE_POW_X86)])
        self.assertEqual(findings, [])

    def test_known_exception_when_macros(self):
        rule = {
            "id": "r",
            "forbid_undefined": ["@soft_double"],
            "known_exceptions": {
                "logf": {
                    "symbols": ["@soft_double"],
                    "when_macros": ["!__ARM_FP&0x8"],
                    "note": "being fixed",
                }
            },
        }
        v7em = make_obj("logf", NM_LOGF_ARM, macros=lsc.parse_macros(MACROS_ARMV7EM))
        findings, _ = self.evaluate([rule], [v7em])
        self.assertEqual(statuses(findings), [])
        self.assertEqual(len(statuses(findings, lsc.STATUS_KNOWN)), 2)
        # On a target with hardware double the exception does not apply, so a
        # soft-double reference is a failure.
        v81m = make_obj("logf", NM_LOGF_ARM, macros=lsc.parse_macros(MACROS_ARMV81M))
        findings, _ = self.evaluate([rule], [v81m])
        self.assertEqual(len(statuses(findings)), 2)
        # A clean object on a target the exception does not apply to gets no
        # stale exception note; on a target it applies to, it does.
        clean = "00000000 00000010 T logf\n"
        v81m = make_obj("logf", clean, macros=lsc.parse_macros(MACROS_ARMV81M))
        findings, _ = self.evaluate([rule], [v81m])
        self.assertEqual(findings, [])
        v7em = make_obj("logf", clean, macros=lsc.parse_macros(MACROS_ARMV7EM))
        findings, _ = self.evaluate([rule], [v7em])
        self.assertEqual(statuses(findings, lsc.STATUS_NOTE), [("r", "logf")])

    def test_unknown_alias(self):
        rule = {"id": "r", "forbid_undefined": ["@missing"]}
        with self.assertRaises(ValueError):
            self.evaluate([rule], [make_obj("logf", NM_LOGF_ARM)])


class ReportTest(unittest.TestCase):
    def test_text_and_markdown(self):
        rule = {"id": "r", "forbid_defined": ["common_constants_internal::LOG_R"]}
        objs = [make_obj("logf", NM_LOGF_ARM)]
        ctx = lsc.RuleContext(set(), {"logf"})
        findings, summary = lsc.evaluate({"rules": [rule]}, objs, ctx)
        text = lsc.format_text(findings, summary)
        self.assertIn("FAIL: [r] logf: defines forbidden symbol", text)
        self.assertIn("1 failures", text)
        md = lsc.format_markdown(findings, summary, objs)
        self.assertIn("| r | yes | 1 | 1 | 0 | 0 |", md)
        self.assertIn("| logf |", md)


class RulesFileTest(unittest.TestCase):
    """Sanity checks of the rules files in this directory."""

    KNOWN_KEYS = {
        "id",
        "comment",
        "functions",
        "exclude_functions",
        "exclude_variants",
        "exceptions",
        "known_exceptions",
        "when_flags",
        "when_macros",
        "forbid_defined",
        "forbid_defined_regex",
        "forbid_undefined",
        "require_defined",
        "max_named_data_object_bytes",
        "max_named_data_object_exceptions",
        "max_writable_data_bytes",
        "allow_writable_sections",
    }
    KNOWN_EXCEPTION_KEYS = {"symbols", "note", "when_macros"}

    def test_rules_files(self):
        import json

        rules_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")
        if not os.path.isdir(rules_dir):
            self.skipTest("no rules directory")
        for name in sorted(os.listdir(rules_dir)):
            if not name.endswith(".json"):
                continue
            with self.subTest(rules=name):
                with open(os.path.join(rules_dir, name), encoding="utf-8") as f:
                    doc = json.load(f)
                ids = [r["id"] for r in doc["rules"]]
                self.assertEqual(len(ids), len(set(ids)), "duplicate rule ids")
                aliases = doc.get("regex_aliases", {})
                for alias in aliases.values():
                    re.compile(alias)
                for rule in doc["rules"]:
                    unknown = set(rule) - self.KNOWN_KEYS
                    self.assertFalse(unknown, "%s: unknown keys" % rule["id"])
                    for entry in rule.get("known_exceptions", {}).values():
                        unknown = set(entry) - self.KNOWN_EXCEPTION_KEYS
                        self.assertFalse(unknown, "%s: unknown keys" % rule["id"])
                    for p in rule.get("functions", []):
                        re.compile(p)
                    for key in ("forbid_defined", "forbid_defined_regex"):
                        for p in rule.get(key, []):
                            re.compile(aliases.get(p[1:], p) if p[0] == "@" else p)
                    for p in rule.get("forbid_undefined", []):
                        if p.startswith("@"):
                            self.assertIn(p[1:], aliases)
                # Exercise every rule on an empty object to catch typos in
                # patterns and aliases.
                ctx = lsc.RuleContext(
                    lsc.parse_math_flags("LIBC_MATH_FAST|LIBC_MATH_SMALL_TABLES"),
                    set(),
                    aliases,
                )
                for rule in doc["rules"]:
                    lsc.evaluate_rule(rule, lsc.ObjectInfo("x", [], [], set(), {}), ctx)


if __name__ == "__main__":
    unittest.main()
