# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from pprint import pformat
import ast
import re
import shutil
import subprocess
import sys


def read_syms_from_list(slist):
    """
    Read a list of symbols from a list of strings.
    Each string is one symbol.
    """
    return [ast.literal_eval(l) for l in slist]


def read_syms_from_file(filename):
    """
    Read a list of symbols in from a file.
    """
    with open(filename, "r") as f:
        data = f.read()
    return read_syms_from_list(data.splitlines())


def read_exclusions(filename):
    with open(filename, "r") as f:
        data = f.read()
    lines = [l.strip() for l in data.splitlines() if l.strip()]
    lines = [l for l in lines if not l.startswith("#")]
    return lines


def write_syms(sym_list, out=None, names_only=False, filter=None):
    """
    Write a list of symbols to the file named by out.
    """
    out_str = ""
    out_list = sym_list
    out_list.sort(key=lambda x: x["name"])
    if filter is not None:
        out_list = filter(out_list)
    if names_only:
        out_list = [sym["name"] for sym in out_list]
    for sym in out_list:
        # Use pformat for consistent ordering of keys.
        out_str += pformat(sym, width=100000) + "\n"
    if out is None:
        sys.stdout.write(out_str)
    else:
        with open(out, "w") as f:
            f.write(out_str)


_cppfilt_exe = shutil.which("c++filt")


def demangle_symbol(symbol):
    if _cppfilt_exe is None:
        return symbol
    result = subprocess.run([_cppfilt_exe], input=symbol.encode(), capture_output=True)
    if result.returncode != 0:
        return symbol
    return result.stdout.decode()


def is_elf(filename):
    with open(filename, "rb") as f:
        magic_bytes = f.read(4)
    return magic_bytes == b"\x7fELF"


def is_mach_o(filename):
    with open(filename, "rb") as f:
        magic_bytes = f.read(4)
    return magic_bytes in [
        b"\xfe\xed\xfa\xce",  # MH_MAGIC
        b"\xce\xfa\xed\xfe",  # MH_CIGAM
        b"\xfe\xed\xfa\xcf",  # MH_MAGIC_64
        b"\xcf\xfa\xed\xfe",  # MH_CIGAM_64
        b"\xca\xfe\xba\xbe",  # FAT_MAGIC
        b"\xbe\xba\xfe\xca",  # FAT_CIGAM
    ]


def is_xcoff_or_big_ar(filename):
    with open(filename, "rb") as f:
        magic_bytes = f.read(7)
    return (
        magic_bytes[:4] in [b"\x01DF", b"\x01F7"]  # XCOFF32  # XCOFF64
        or magic_bytes == b"<bigaf>"
    )


def is_library_file(filename):
    if sys.platform == "darwin":
        return is_mach_o(filename)
    elif sys.platform.startswith("aix"):
        return is_xcoff_or_big_ar(filename)
    else:
        return is_elf(filename)


def extract_or_load(filename):
    import libcxx.sym_check.extract

    if is_library_file(filename):
        return libcxx.sym_check.extract.extract_symbols(filename)
    return read_syms_from_file(filename)


def adjust_mangled_name(name):
    if not name.startswith("__Z"):
        return name
    return name[1:]


def extract_object_sizes_from_map(mapfilename: str):
    maplines = []
    result = {}

    try:
        with open(mapfilename, "r") as f:
            maplines = f.readlines()
    except FileNotFoundError as e:
        return False, {}

    if len(maplines) == 0:
        return False, {}

    # Yes, this is fragile.
    symbols_start_index = -1
    for i in range(0, len(maplines)):
        if re.search("# Symbols:", maplines[i]):
            symbols_start_index = i
            break

    if symbols_start_index == -1:
        return False, {}

    # There is a header after the line we use to detect
    # the start of the symbol section -- + 2 to jump
    # over that.
    maplines = maplines[symbols_start_index + 2 :]
    for line in maplines:
        components = line.split()
        if len(components) != 5:
            continue
        name = components[4]
        size = int(components[1], 16)
        if size != 0:
            result[name] = size

    return True, result


def update_symbols_with_supplemental_information(symbols, supplemental_info):
    for sym in symbols:
        # Only update from the supplementatl information where the symbol has a
        # size, that size is not 0 and its type is OBJECT.
        if "size" not in sym or sym["size"] != 0 or sym["type"] != "OBJECT":
            continue
        if sym["name"] in supplemental_info:
            updated_size = supplemental_info[sym["name"]]
            sym["size"] = updated_size


new_delete_std_symbols = ["_Znam", "_Znwm", "_ZdaPv", "_ZdaPvm", "_ZdlPv", "_ZdlPvm"]

cxxabi_symbols = [
    "___dynamic_cast",
    "___gxx_personality_v0",
    "_ZTIDi",
    "_ZTIDn",
    "_ZTIDs",
    "_ZTIPDi",
    "_ZTIPDn",
    "_ZTIPDs",
    "_ZTIPKDi",
    "_ZTIPKDn",
    "_ZTIPKDs",
    "_ZTIPKa",
    "_ZTIPKb",
    "_ZTIPKc",
    "_ZTIPKd",
    "_ZTIPKe",
    "_ZTIPKf",
    "_ZTIPKh",
    "_ZTIPKi",
    "_ZTIPKj",
    "_ZTIPKl",
    "_ZTIPKm",
    "_ZTIPKs",
    "_ZTIPKt",
    "_ZTIPKv",
    "_ZTIPKw",
    "_ZTIPKx",
    "_ZTIPKy",
    "_ZTIPa",
    "_ZTIPb",
    "_ZTIPc",
    "_ZTIPd",
    "_ZTIPe",
    "_ZTIPf",
    "_ZTIPh",
    "_ZTIPi",
    "_ZTIPj",
    "_ZTIPl",
    "_ZTIPm",
    "_ZTIPs",
    "_ZTIPt",
    "_ZTIPv",
    "_ZTIPw",
    "_ZTIPx",
    "_ZTIPy",
    "_ZTIa",
    "_ZTIb",
    "_ZTIc",
    "_ZTId",
    "_ZTIe",
    "_ZTIf",
    "_ZTIh",
    "_ZTIi",
    "_ZTIj",
    "_ZTIl",
    "_ZTIm",
    "_ZTIs",
    "_ZTIt",
    "_ZTIv",
    "_ZTIw",
    "_ZTIx",
    "_ZTIy",
    "_ZTSDi",
    "_ZTSDn",
    "_ZTSDs",
    "_ZTSPDi",
    "_ZTSPDn",
    "_ZTSPDs",
    "_ZTSPKDi",
    "_ZTSPKDn",
    "_ZTSPKDs",
    "_ZTSPKa",
    "_ZTSPKb",
    "_ZTSPKc",
    "_ZTSPKd",
    "_ZTSPKe",
    "_ZTSPKf",
    "_ZTSPKh",
    "_ZTSPKi",
    "_ZTSPKj",
    "_ZTSPKl",
    "_ZTSPKm",
    "_ZTSPKs",
    "_ZTSPKt",
    "_ZTSPKv",
    "_ZTSPKw",
    "_ZTSPKx",
    "_ZTSPKy",
    "_ZTSPa",
    "_ZTSPb",
    "_ZTSPc",
    "_ZTSPd",
    "_ZTSPe",
    "_ZTSPf",
    "_ZTSPh",
    "_ZTSPi",
    "_ZTSPj",
    "_ZTSPl",
    "_ZTSPm",
    "_ZTSPs",
    "_ZTSPt",
    "_ZTSPv",
    "_ZTSPw",
    "_ZTSPx",
    "_ZTSPy",
    "_ZTSa",
    "_ZTSb",
    "_ZTSc",
    "_ZTSd",
    "_ZTSe",
    "_ZTSf",
    "_ZTSh",
    "_ZTSi",
    "_ZTSj",
    "_ZTSl",
    "_ZTSm",
    "_ZTSs",
    "_ZTSt",
    "_ZTSv",
    "_ZTSw",
    "_ZTSx",
    "_ZTSy",
]


def is_stdlib_symbol_name(name, sym):
    name = adjust_mangled_name(name)
    if re.search("@GLIBC|@GCC", name):
        # Only when symbol is defined do we consider it ours
        return sym["is_defined"]
    if re.search("(St[0-9])|(__cxa)|(__cxxabi)", name):
        return True
    if name in new_delete_std_symbols:
        return True
    if name in cxxabi_symbols:
        return True
    if name.startswith("_Z"):
        return True
    return False


def filter_stdlib_symbols(syms):
    stdlib_symbols = []
    other_symbols = []
    for s in syms:
        canon_name = adjust_mangled_name(s["name"])
        if not is_stdlib_symbol_name(canon_name, s):
            other_symbols += [s]
        else:
            stdlib_symbols += [s]
    return stdlib_symbols, other_symbols
