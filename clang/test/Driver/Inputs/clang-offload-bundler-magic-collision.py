#!/usr/bin/env python3
"""Regenerate clang-offload-bundler-magic-collision.co.

This input reproduces a magic-byte collision in the compressed offload bundle
(CCOB) reader. The reader used to find the boundary between concatenated
bundles by scanning for the literal 4-byte magic "CCOB"; because that magic can
appear by chance inside a compressed payload, the reader could truncate a valid
single bundle and then fail to decompress it.

To make the collision deterministic we take a real compressed bundle produced
by clang-offload-bundler and splice a zstd *skippable frame* whose body is the
bytes "CCOB" into the compressed region. A zstd skippable frame is ignored by
the decompressor, so the bundle still decompresses correctly, but a naive
find("CCOB", 4) scan stops at the planted bytes.

The bundle's header FileSize is updated so it remains authoritative. A reader
that advances by FileSize handles this file correctly; a reader that scans for
"CCOB" truncates and fails.

Usage:
  clang-offload-bundler-magic-collision.py <clang-offload-bundler> [output.co]
"""

import os
import struct
import subprocess
import sys
import tempfile

# zstd skippable frame magic numbers are 0x184D2A50..0x184D2A5F.
ZSTD_SKIPPABLE_MAGIC = 0x184D2A50
PLANTED_MAGIC = b"CCOB"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: %s <clang-offload-bundler> [output.co]" % sys.argv[0])
    bundler = sys.argv[1]
    out = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "clang-offload-bundler-magic-collision.co",
        )
    )

    with tempfile.TemporaryDirectory() as d:
        dev1 = os.path.join(d, "dev1")
        dev2 = os.path.join(d, "dev2")
        with open(dev1, "wb") as f:
            f.write(b"Content of device file 1\n")
        with open(dev2, "wb") as f:
            f.write(b"Content of device file 2\n")
        base = os.path.join(d, "base.co")
        subprocess.run(
            [
                bundler,
                "-compress",
                "-type=bc",
                "-targets=hip-amdgcn-amd-amdhsa--gfx906,"
                "hip-amdgcn-amd-amdhsa--gfx908",
                "-input=" + dev1,
                "-input=" + dev2,
                "-output=" + base,
            ],
            check=True,
        )
        data = bytearray(open(base, "rb").read())

    if data[:4] != b"CCOB":
        sys.exit("unexpected magic in compressed bundle")
    version = struct.unpack_from("<H", data, 4)[0]
    if version == 2:
        header_size, fs_off, fs_fmt = 24, 8, "<I"
    elif version == 3:
        header_size, fs_off, fs_fmt = 32, 8, "<Q"
    else:
        sys.exit("unsupported compressed bundle version %d" % version)

    file_size = struct.unpack_from(fs_fmt, data, fs_off)[0]
    if file_size != len(data):
        sys.exit("FileSize %d != file length %d" % (file_size, len(data)))

    # Splice a zstd skippable frame carrying "CCOB" right after the header, so
    # the planted magic lands inside the compressed region.
    skip = struct.pack("<II", ZSTD_SKIPPABLE_MAGIC, len(PLANTED_MAGIC)) + PLANTED_MAGIC
    data = data[:header_size] + skip + data[header_size:]
    struct.pack_into(fs_fmt, data, fs_off, file_size + len(skip))

    planted = bytes(data).find(PLANTED_MAGIC, 4)
    if planted != header_size + 8:
        sys.exit("planted magic at unexpected offset %d" % planted)

    with open(out, "wb") as f:
        f.write(data)
    print(
        "wrote %s: version=%d header=%d FileSize=%d planted 'CCOB' at %d"
        % (out, version, header_size, file_size + len(skip), planted)
    )


if __name__ == "__main__":
    main()
