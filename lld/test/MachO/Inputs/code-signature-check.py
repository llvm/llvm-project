"""Checks the validity of MachO binary signatures

MachO binaries sometimes include a LC_CODE_SIGNATURE load command
and corresponding section in the __LINKEDIT segment that together
work to "sign" the binary. This script is used to check the validity
of this signature.

Usage:
    ./code-signature-check.py my_binary 800 300 0 800

Arguments:
   binary - The MachO binary to be tested
   offset - The offset from the start of the binary to where the code signature section begins
   size - The size of the code signature section in the binary
   code_offset - The point in the binary to begin hashing
   code_size - The length starting from code_offset to hash
"""

import argparse
import collections
import hashlib
import itertools
import struct
import sys
import typing


class CodeDirectoryVersion:
    SUPPORTSSCATTER = 0x20100
    SUPPORTSTEAMID = 0x20200
    SUPPORTSCODELIMIT64 = 0x20300
    SUPPORTSEXECSEG = 0x20400


class CodeDirectory:
    @staticmethod
    def make(
        buf: memoryview,
    ) -> typing.Union[
        "CodeDirectoryBase",
        "CodeDirectoryV20100",
        "CodeDirectoryV20200",
        "CodeDirectoryV20300",
        "CodeDirectoryV20400",
    ]:
        _magic, _length, version = struct.unpack_from(">III", buf, 0)
        subtype = {
            CodeDirectoryVersion.SUPPORTSSCATTER: CodeDirectoryV20100,
            CodeDirectoryVersion.SUPPORTSTEAMID: CodeDirectoryV20200,
            CodeDirectoryVersion.SUPPORTSCODELIMIT64: CodeDirectoryV20300,
            CodeDirectoryVersion.SUPPORTSEXECSEG: CodeDirectoryV20400,
        }.get(version, CodeDirectoryBase)

        return subtype._make(struct.unpack_from(subtype._format(), buf, 0))


class CodeDirectoryBase(typing.NamedTuple):
    magic: int
    length: int
    version: int
    flags: int
    hashOffset: int
    identOffset: int
    nSpecialSlots: int
    nCodeSlots: int
    codeLimit: int
    hashSize: int
    hashType: int
    platform: int
    pageSize: int
    spare2: int

    @staticmethod
    def _format() -> str:
        return ">IIIIIIIIIBBBBI"


class CodeDirectoryV20100(typing.NamedTuple):
    magic: int
    length: int
    version: int
    flags: int
    hashOffset: int
    identOffset: int
    nSpecialSlots: int
    nCodeSlots: int
    codeLimit: int
    hashSize: int
    hashType: int
    platform: int
    pageSize: int
    spare2: int

    scatterOffset: int

    @staticmethod
    def _format() -> str:
        return CodeDirectoryBase._format() + "I"


class CodeDirectoryV20200(typing.NamedTuple):
    magic: int
    length: int
    version: int
    flags: int
    hashOffset: int
    identOffset: int
    nSpecialSlots: int
    nCodeSlots: int
    codeLimit: int
    hashSize: int
    hashType: int
    platform: int
    pageSize: int
    spare2: int

    scatterOffset: int

    teamOffset: int

    @staticmethod
    def _format() -> str:
        return CodeDirectoryV20100._format() + "I"


class CodeDirectoryV20300(typing.NamedTuple):
    magic: int
    length: int
    version: int
    flags: int
    hashOffset: int
    identOffset: int
    nSpecialSlots: int
    nCodeSlots: int
    codeLimit: int
    hashSize: int
    hashType: int
    platform: int
    pageSize: int
    spare2: int

    scatterOffset: int

    teamOffset: int

    spare3: int
    codeLimit64: int

    @staticmethod
    def _format() -> str:
        return CodeDirectoryV20200._format() + "IQ"


class CodeDirectoryV20400(typing.NamedTuple):
    magic: int
    length: int
    version: int
    flags: int
    hashOffset: int
    identOffset: int
    nSpecialSlots: int
    nCodeSlots: int
    codeLimit: int
    hashSize: int
    hashType: int
    platform: int
    pageSize: int
    spare2: int

    scatterOffset: int

    teamOffset: int

    spare3: int
    codeLimit64: int

    execSegBase: int
    execSegLimit: int
    execSegFlags: int

    @staticmethod
    def _format() -> str:
        return CodeDirectoryV20300._format() + "QQQ"


class CodeDirectoryBlobIndex(typing.NamedTuple):
    type_: int
    offset: int

    @staticmethod
    def make(buf: memoryview) -> "CodeDirectoryBlobIndex":
        return CodeDirectoryBlobIndex._make(
            struct.unpack_from(CodeDirectoryBlobIndex.__format(), buf, 0)
        )

    @staticmethod
    def bytesize() -> int:
        return struct.calcsize(CodeDirectoryBlobIndex.__format())

    @staticmethod
    def __format() -> str:
        return ">II"


class CodeDirectorySuperBlob(typing.NamedTuple):
    magic: int
    length: int
    count: int
    blob_indices: typing.List[CodeDirectoryBlobIndex]

    @staticmethod
    def make(buf: memoryview) -> "CodeDirectorySuperBlob":
        super_blob_layout = ">III"
        super_blob = struct.unpack_from(super_blob_layout, buf, 0)

        offset = struct.calcsize(super_blob_layout)
        blob_indices = []
        for idx in range(super_blob[2]):
            blob_indices.append(CodeDirectoryBlobIndex.make(buf[offset:]))
            offset += CodeDirectoryBlobIndex.bytesize()

        return CodeDirectorySuperBlob(*super_blob, blob_indices)


def unpack_null_terminated_string(buf: memoryview) -> str:
    b = bytes(itertools.takewhile(lambda b: b != 0, buf))
    return b.decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "binary", type=argparse.FileType("rb"), help="The file to analyze"
    )
    parser.add_argument(
        "offset", type=int, help="Offset to start of Code Directory data"
    )
    parser.add_argument("size", type=int, help="Size of Code Directory data")
    parser.add_argument(
        "code_offset", type=int, help="Offset to start of code pages to hash"
    )
    parser.add_argument("code_size", type=int, help="Size of the code pages to hash")

    args = parser.parse_args()

    args.binary.seek(args.offset)
    super_blob_bytes = args.binary.read(args.size)
    super_blob_mem = memoryview(super_blob_bytes)

    super_blob = CodeDirectorySuperBlob.make(super_blob_mem)
    print(super_blob)

    for blob_index in super_blob.blob_indices:
        code_directory_offset = blob_index.offset
        code_directory = CodeDirectory.make(super_blob_mem[code_directory_offset:])
        print(code_directory)

        ident_offset = code_directory_offset + code_directory.identOffset
        print(
            "Code Directory ID: "
            + unpack_null_terminated_string(super_blob_mem[ident_offset:])
        )

        code_offset = args.code_offset
        code_end = code_offset + args.code_size
        page_size = 1 << code_directory.pageSize
        args.binary.seek(code_offset)

        hashes_offset = code_directory_offset + code_directory.hashOffset
        for idx in range(code_directory.nCodeSlots):
            hash_bytes = bytes(
                super_blob_mem[hashes_offset : hashes_offset + code_directory.hashSize]
            )
            hashes_offset += code_directory.hashSize

            hasher = hashlib.sha256()
            read_size = min(page_size, code_end - code_offset)
            hasher.update(args.binary.read(read_size))
            calculated_hash_bytes = hasher.digest()
            code_offset += read_size

            print("%s <> %s" % (hash_bytes.hex(), calculated_hash_bytes.hex()))

            if hash_bytes != calculated_hash_bytes:
                sys.exit(-1)


if __name__ == "__main__":
    main()
