#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
"""Unit tests for docgen and header implementation status evaluation."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from docgen import (
    check_api,
    cppreference_url,
    generate_index_md,
    posix_url,
    print_header_status_md,
)
from header import Header, _is_stub_file


class DocgenTest(unittest.TestCase):
    def test_header_stem(self):
        self.assertEqual(Header("math.h").stem, "math")
        self.assertEqual(Header("search.h").stem, "search")
        self.assertEqual(Header("arpa/inet.h").stem, "arpa/inet")

    def test_new_c23_and_posix_headers(self):
        self.assertTrue(Header("limits.h").implements_macro("CHAR_BIT"))
        self.assertFalse(Header("limits.h").implements_macro("BITINT_MAXWIDTH"))
        self.assertTrue(Header("stdint.h").implements_type("int64_t"))
        self.assertTrue(Header("stdint.h").implements_macro("INT64_MAX"))
        self.assertTrue(Header("stdckdint.h").implements_macro("ckd_add"))
        self.assertTrue(Header("regex.h").implements_fn("regcomp"))
        self.assertTrue(Header("sys/shm.h").implements_fn("shmget"))
        self.assertFalse(Header("ulimit.h").implements_fn("ulimit"))

    def test_is_stub_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stub = Path(tmpdir) / "stub.cpp"
            stub.write_text(
                "// TODO\n"
                "LLVM_LIBC_FUNCTION(void *, s, ()) {\n"
                "  libc_errno = ENOSYS;\n"
                "  return nullptr;\n"
                "}\n"
            )
            self.assertTrue(_is_stub_file(stub))

            real = Path(tmpdir) / "real.cpp"
            real.write_text("LLVM_LIBC_FUNCTION(int, r, (int x)) { return x + 1; }\n")
            self.assertFalse(_is_stub_file(real))

    def test_implements_fn(self):
        self.assertTrue(Header("stdio.h").implements_fn("printf"))
        self.assertTrue(Header("netinet/in.h").implements_fn("htonl"))
        self.assertTrue(Header("math.h").implements_fn("acospif"))
        self.assertFalse(Header("math.h").implements_fn("tgamma"))
        self.assertFalse(Header("stdio.h").implements_fn("sqrt"))
        self.assertFalse(Header("math.h").implements_fn("printf"))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src" / "foo"
            cfg = root / "config" / "linux"
            src.mkdir(parents=True)
            cfg.mkdir(parents=True)
            (src / "stub_fn.cpp").write_text(
                "// TODO\nLLVM_LIBC_FUNCTION(int, stub_fn, ()) { return -1; }\n"
            )
            (src / "real_fn.cpp").write_text(
                "LLVM_LIBC_FUNCTION(int, real_fn, (int x)) { return x + 1; }\n"
            )
            (cfg / "entrypoints.txt").write_text(
                "libc.src.foo.stub_fn\nlibc.src.foo.real_fn\n"
            )
            hdr = Header("foo.h", libc_root=root)
            self.assertFalse(hdr.implements_fn("stub_fn"))
            self.assertTrue(hdr.implements_fn("real_fn"))

    def test_implements_macro(self):
        self.assertTrue(Header("assert.h").implements_macro("assert"))
        self.assertTrue(Header("math.h").implements_macro("isnan"))
        self.assertFalse(Header("math.h").implements_macro("isgreater"))
        self.assertFalse(Header("sys/syscall.h").macro_file_exists())
        self.assertTrue(Header("sys/syscall.h").implements_macro("SYS_read"))
        self.assertTrue(Header("errno.h").implements_macro("errno"))
        self.assertTrue(Header("unistd.h").implements_macro("SEEK_SET"))
        self.assertTrue(Header("float.h").implements_macro("FLT_MAX"))
        self.assertFalse(Header("float.h").implements_macro("NAN"))

    def test_implements_type(self):
        self.assertTrue(Header("stdio.h").implements_type("FILE"))
        self.assertTrue(Header("time.h").implements_type("struct tm"))
        self.assertTrue(Header("setjmp.h").implements_type("sigjmp_buf"))
        self.assertFalse(Header("wctype.h").implements_type("wctrans_t"))
        self.assertFalse(Header("monetary.h").implements_type("size_t"))

    def test_implements_object(self):
        self.assertTrue(Header("stdio.h").implements_object("stdin"))
        self.assertTrue(Header("time.h").implements_object("tzname"))
        self.assertTrue(Header("unistd.h").implements_object("optarg"))
        self.assertFalse(Header("time.h").implements_object("getdate_err"))
        self.assertFalse(Header("errno.h").implements_object("errno"))

    def test_md_output_generation(self):
        hdr = Header("stdio.h")
        api = {
            "c-reference-root": "io",
            "types": {
                "FILE": {
                    "c-definition": "7.23.1",
                    "c-reference": "io/FILE",
                    "in-latest-posix": None,
                }
            },
            "objects": {
                "stdin": {
                    "c-definition": "7.23.1",
                    "c-reference": "io/std_streams",
                    "in-latest-posix": None,
                }
            },
            "macros": {"EOF": {"c-definition": "7.23.1", "in-latest-posix": None}},
            "functions": {
                "printf": {
                    "c-definition": "7.23.6.3",
                    "c-reference": "io/fprintf",
                    "in-latest-posix": None,
                },
                "printf_s": {
                    "c-definition": "K.3.5.3.3",
                    "c-reference": "io/fprintf",
                },
                "missing_fn": {"c-definition": "7.23.6.99"},
                "gnu_fn": {"c-definition": "TODO: glibc extension"},
            },
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_header_status_md(hdr, api)
        out = buf.getvalue()
        lines = {line.strip() for line in out.splitlines()}
        self.assertIn("# stdio.h", out)
        self.assertIn("## Types", out)
        self.assertIn("## Global Variables", out)
        self.assertIn("## Macros", out)
        self.assertIn("## Functions", out)
        self.assertIn('<table class="hc-table">', out)
        self.assertTrue(
            any(
                'id="type-FILE"' in l
                and 'title="Implemented"' in l
                and 'href="https://en.cppreference.com/w/c/io/FILE"' in l
                for l in lines
            )
        )
        self.assertTrue(
            any(
                'id="object-stdin"' in l
                and 'title="Implemented"' in l
                and 'href="https://en.cppreference.com/w/c/io/std_streams"' in l
                for l in lines
            )
        )
        self.assertTrue(
            any(
                'id="macro-EOF"' in l
                and 'title="Implemented"' in l
                and 'href="https://en.cppreference.com/w/c/io"' in l
                and "basedefs/stdio.h.html" in l
                and '<span aria-hidden="true">&#8599;</span>' in l
                for l in lines
            )
        )
        self.assertTrue(
            any(
                'id="function-printf"' in l
                and 'title="Implemented"' in l
                and 'href="https://en.cppreference.com/w/c/io/fprintf"' in l
                and "functions/printf.html" in l
                for l in lines
            )
        )
        self.assertTrue(
            any(
                'id="function-printf_s"' in l
                and 'href="https://en.cppreference.com/w/c/io/fprintf"' in l
                for l in lines
            )
        )
        self.assertTrue(
            any(
                'id="function-missing_fn"' in l
                and 'title="Not Implemented"' in l
                and 'href="https://en.cppreference.com/w/c/io/missing_fn"' in l
                for l in lines
            )
        )
        self.assertTrue(
            any(
                'id="function-gnu_fn"' in l
                and '<span class="hc-sec-badge">TODO: glibc extension</span>' in l
                for l in lines
            )
        )

    def test_external_urls(self):
        cases = [
            (
                "stdio.h",
                "fopen",
                "function",
                "functions",
                "https://en.cppreference.com/w/c/io/fopen",
            ),
            (
                "stdio.h",
                "printf",
                "function",
                "functions",
                "https://en.cppreference.com/w/c/io/fprintf",
            ),
            (
                "stdio.h",
                "SEEK_SET",
                "macro",
                "macros",
                "https://en.cppreference.com/w/c/io/fseek",
            ),
            (
                "stdlib.h",
                "malloc",
                "function",
                "functions",
                "https://en.cppreference.com/w/c/memory/malloc",
            ),
            (
                "stdlib.h",
                "strfromd128",
                "function",
                "functions",
                "https://en.cppreference.com/w/c/string/byte/strfromf",
            ),
            (
                "stdbit.h",
                "stdc_bit_ceil_ui",
                "function",
                "functions",
                "https://en.cppreference.com/w/c/numeric/bit/stdc_bit_ceil",
            ),
            (
                "time.h",
                "clock",
                "function",
                "functions",
                "https://en.cppreference.com/w/c/chrono/clock",
            ),
            ("unistd.h", "fork", "function", "functions", ""),
        ]
        for hname, sym, kind, sec, expected in cases:
            hdr = Header(hname)
            item = (hdr.api.get(sec) or {}).get(sym, {})
            with self.subTest(header=hname, symbol=sym):
                self.assertEqual(cppreference_url(hdr, sym, kind, item), expected)
        with redirect_stderr(io.StringIO()):
            for yf in sorted(Path(__file__).parent.rglob("*.yaml")):
                rel = yf.relative_to(Path(__file__).parent).with_suffix(".h").as_posix()
                hdr = Header(rel)
                check_api(hdr, hdr.api)
        self.assertEqual(
            posix_url(
                Header("sys/stat.h"), "chmod", "function", {"in-latest-posix": None}
            ),
            "https://pubs.opengroup.org/onlinepubs/9799919799/functions/chmod.html",
        )
        self.assertEqual(
            posix_url(
                Header("sys/stat.h"), "S_IRUSR", "macro", {"in-latest-posix": None}
            ),
            "https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/sys_stat.h.html",
        )
        self.assertEqual(
            posix_url(
                Header("strings.h"), "bcmp", "function", {"removed-in-posix-2008": None}
            ),
            "https://pubs.opengroup.org/onlinepubs/007904875/functions/bcmp.html",
        )
        self.assertEqual(
            posix_url(
                Header("sys/time.h"),
                "gettimeofday",
                "function",
                {"removed-in-posix-2024": None},
            ),
            "https://pubs.opengroup.org/onlinepubs/"
            "9699919799.2018edition/functions/gettimeofday.html",
        )
        self.assertEqual(posix_url(Header("stdio.h"), "printf", "function", {}), "")

    def test_families_md(self):
        math_hdr = Header("math.h")
        ev = math_hdr.evaluate_family("atan2", math_hdr.api["families"]["atan2"])
        self.assertTrue(ev["implemented"])
        self.assertTrue(ev["variants"]["float"]["implemented"])
        self.assertEqual(ev["variants"]["double"]["accuracy"], "1 ULP")
        self.assertFalse(ev["variants"]["double_double"]["implemented"])

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_header_status_md(math_hdr, math_hdr.api)
        math_out = buf.getvalue()
        self.assertIn("{doc}`../math/index`", math_out)
        self.assertIn("## Basic Operations", math_out)
        self.assertIn("## Higher Math Functions", math_out)
        self.assertNotIn("## GPU Conformance", math_out)
        self.assertIn('title="atan2: Implemented (1 ULP)">1 ULP</span>', math_out)
        self.assertIn('title="log_bf16: Implemented"', math_out)
        self.assertIn(
            'class="hc-status-na" title="Not applicable">N/A</span>', math_out
        )
        self.assertIn("F.10.1.4", math_out)

        cmplx_hdr = Header("complex.h")
        cbuf = io.StringIO()
        with redirect_stdout(cbuf):
            print_header_status_md(cmplx_hdr, cmplx_hdr.api)
        cmplx_out = cbuf.getvalue()
        self.assertIn("## Function Families", cmplx_out)
        self.assertIn('title="carg: Implemented (1 ULP)">1 ULP</span>', cmplx_out)
        self.assertIn("G.6.2.1", cmplx_out)

        fx_hdr = Header("stdfix.h")
        fbuf = io.StringIO()
        with redirect_stdout(fbuf):
            print_header_status_md(fx_hdr, fx_hdr.api)
        fx_out = fbuf.getvalue()
        self.assertIn("{doc}`../math/stdfix`", fx_out)
        self.assertIn("## Fixed-point Arithmetics", fx_out)
        self.assertIn("## Higher Math Functions", fx_out)
        self.assertIn("## Conversion Functions", fx_out)
        self.assertIn('title="uhrbits: Implemented"', fx_out)
        self.assertIn('title="exphk: Implemented"', fx_out)
        self.assertIn('title="strtofxr: Not Implemented"', fx_out)

        srch_hdr = Header("search.h")
        self.assertTrue(srch_hdr.implements_type("ACTION"))
        self.assertTrue(srch_hdr.implements_fn("hcreate"))

    def test_summary_stats(self):
        stdio = Header("stdio.h").get_summary_stats()
        self.assertEqual(stdio["standard"], "c23")
        self.assertGreater(stdio["c23"]["implemented"], 0)
        self.assertEqual(stdio["types"]["implemented"], stdio["types"]["total"])
        self.assertEqual(stdio["objects"]["implemented"], 3)
        for key in ("overall", "functions", "macros", "types", "objects", "c23"):
            self.assertLessEqual(stdio[key]["implemented"], stdio[key]["total"])

        self.assertEqual(Header("fcntl.h").get_summary_stats()["c23"]["total"], 0)
        self.assertTrue(Header("ulimit.h").get_summary_stats()["removed_posix"])
        self.assertFalse(Header("byteswap.h").get_summary_stats()["removed_posix"])

        math_hdr = Header("math.h")
        math_stats = math_hdr.get_summary_stats()
        self.assertEqual(len(math_stats["precisions"]), 7)
        self.assertEqual(
            math_stats["overall"]["total"],
            sum(
                math_stats[k]["total"]
                for k in ("functions", "macros", "types", "objects")
            ),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_header_status_md(math_hdr, math_hdr.api)
        out = buf.getvalue()
        self.assertIn('role="progressbar"', out)
        self.assertIn("Base Families", out)
        self.assertIn("Global Variables", out)
        self.assertIn("Precision Coverage", out)

        fx_stats = Header("stdfix.h").get_summary_stats()
        self.assertEqual(len(fx_stats["precisions"]), 12)
        self.assertGreater(fx_stats["tr18037"]["implemented"], 0)
        self.assertEqual(fx_stats["c23"]["total"], 0)

    def test_index_output_generation(self):
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            generate_index_md()
        out = buf.getvalue()
        self.assertIn("(implementation_status)=", out)
        self.assertIn("(headers_index)=", out)
        self.assertIn("# Headers\n", out)
        self.assertIn("C23 Standard Completeness", out)
        self.assertIn("POSIX.1-2024 Completeness", out)
        self.assertIn("Global Variables", out)
        self.assertIn("`stdarg.h`", out)
        self.assertIn('data-stem="stdio"', out)
        self.assertIn('data-stem="arpa/inet"', out)
        self.assertIn("acosf128", out)
        self.assertIn('class="dashboard-controls"', out)
        self.assertIn('class="hc-chip-missing"', out)
        self.assertIn('class="hc-chip-impl"', out)
        self.assertIn('href="https://en.cppreference.com/w/c/io/clearerr"', out)
        self.assertIn("```{toctree}\n:hidden: true", out)
        self.assertIn("\npoll\npthread\npwd\n", out)


if __name__ == "__main__":
    unittest.main()
