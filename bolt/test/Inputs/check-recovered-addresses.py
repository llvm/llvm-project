# Validate recovered addresses by reading the input and output ELF files rather
# than matching llvm-bolt diagnostics. All modes verify that target moved and
# that the ELF entry points to the recovered _start. The AArch64 mode decodes
# the ADRP/ADD address in _start. The x86 modes check a data pointer and the
# .init_array entry, including the pointer left unchanged in conservative mode.
# Usage: check-recovered-addresses.py {aarch64,x86,x86-conservative} input output
import struct
import sys


class ELF:
    def __init__(self, path):
        self.data = open(path, "rb").read()
        assert self.data[:6] == b"\x7fELF\x02\x01", "expected ELF64LE"
        header = struct.unpack_from("<HHIQQQIHHHHHH", self.data, 16)
        self.entry = header[3]
        shoff, shsize, shnum, shstr = header[5], header[10], header[11], header[12]
        self.sections = [
            struct.unpack_from("<IIQQQQIIQQ", self.data, shoff + i * shsize)
            for i in range(shnum)
        ]
        names = self.contents(self.sections[shstr])
        self.named_sections = {self.string(names, s[0]): s for s in self.sections}
        self.symbols = {}
        for section in self.sections:
            if section[1] != 2:  # SHT_SYMTAB
                continue
            strings = self.contents(self.sections[section[6]])
            for offset in range(section[4], section[4] + section[5], section[9]):
                name, info, other, index, value, size = struct.unpack_from(
                    "<IBBHQQ", self.data, offset
                )
                self.symbols[self.string(strings, name)] = value

    @staticmethod
    def string(data, offset):
        return data[offset : data.index(b"\0", offset)].decode()

    def contents(self, section):
        return self.data[section[4] : section[4] + section[5]]

    def at(self, address, size):
        for section in self.sections:
            if section[2] & 2 and section[1] != 8:
                start = address - section[3]
                if 0 <= start and start + size <= section[5]:
                    return self.contents(section)[start : start + size]
        raise AssertionError(f"unmapped address {address:#x}")


arch, before, after = sys.argv[1:]
old, new = ELF(before), ELF(after)
target = new.symbols["target"]
assert target != old.symbols["target"], "test must move target"
assert new.entry == new.symbols["_start"], "ELF entry must follow recovered start"
if arch == "aarch64":
    start = new.symbols["_start"]
    adrp, add = struct.unpack("<II", new.at(start, 8))
    assert adrp & 0x9F000000 == 0x90000000, "expected ADRP"
    assert add & 0xFF000000 == 0x91000000, "expected ADD immediate"
    immediate = ((adrp >> 29) & 3) | (((adrp >> 5) & 0x7FFFF) << 2)
    if immediate & (1 << 20):
        immediate -= 1 << 21
    address = (start & ~4095) + (immediate << 12) + ((add >> 10) & 4095)
    assert address == target, (hex(address), hex(target))
elif arch in ("x86", "x86-conservative"):
    pointer = struct.unpack("<Q", new.at(new.symbols["pointer"], 8))[0]
    array = struct.unpack("<Q", new.contents(new.named_sections[".init_array"])[:8])[0]
    if arch == "x86":
        assert pointer == target, (hex(pointer), hex(target))
    else:
        assert pointer == old.symbols["target"], (
            hex(pointer),
            hex(old.symbols["target"]),
        )
    assert array == target, (hex(array), hex(target))
else:
    raise AssertionError(f"unexpected architecture: {arch}")
