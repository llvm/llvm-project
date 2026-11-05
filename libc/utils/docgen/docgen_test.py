#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
"""Unit tests for docgen and header implementation status evaluation."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from header import Header, _is_stub_file


class DocgenTest(unittest.TestCase):
    def test_header_stem(self):
        self.assertEqual(Header("math.h").stem, "math")
        self.assertEqual(Header("search.h").stem, "search")
        self.assertEqual(Header("arpa/inet.h").stem, "arpa/inet")

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
        self.assertFalse(Header("sys/syscall.h").macro_file_exists())
        self.assertTrue(Header("sys/syscall.h").implements_macro("SYS_read"))
        self.assertTrue(Header("errno.h").implements_macro("errno"))
        self.assertTrue(Header("unistd.h").implements_macro("SEEK_SET"))
        self.assertTrue(Header("float.h").implements_macro("NAN"))


if __name__ == "__main__":
    unittest.main()
