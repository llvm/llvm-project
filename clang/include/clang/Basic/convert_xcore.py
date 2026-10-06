#!/usr/bin/env python3

import re
import sys
from typing import List, Tuple, Optional


class XCoreConverter:
    def __init__(self):
        self.base_types = {
            "v": "void",
            "b": "bool",
            "c": "char",
            "s": "short",
            "i": "int",
            "h": "__fp16",
            "x": "_Float16",
            "y": "__bf16",
            "f": "float",
            "d": "double",
            "z": "size_t",
            "w": "wchar_t",
            "F": "CFString",
            "G": "id",
            "H": "SEL",
            "M": "struct objc_super",
            "a": "__builtin_va_list",
            "A": "__builtin_va_list&",
            "Y": "ptrdiff_t",
            "P": "FILE*",
            "J": "jmp_buf",
            "p": "pid_t",
        }

        self.attributes = {
            "n": "NoThrow",
            "r": "NoReturn",
            "U": "Pure",
            "c": "Const",
            "t": "CustomTypeChecking",
            "T": "TypeGeneric",
            "F": "LibBuiltin",
            "f": "LibFunction",
            "h": "RequiresHeader",
            "i": "RuntimeLibFunction",
            "e": "ConstWithoutErrnoAndExceptions",
            "g": "ConstWithoutExceptions",
            "j": "ReturnsTwice",
            "u": "NoSideEffects",
            "z": "CXXNamespaceStd",
            "E": "ConstantEvaluated",
            "G": "CXXConsteval",
        }

    def parse_builtin_line(self, line: str) -> Optional[Tuple[str, str, str]]:
        # XCore uses BUILTIN instead of TARGET_BUILTIN
        pattern = r'BUILTIN\(([^,]+),\s*"([^"]*)",\s*"([^"]*)"\)'
        match = re.match(pattern, line.strip())
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None

    def parse_type_encoding(self, encoding: str) -> Tuple[str, List[str]]:
        if not encoding:
            return "void", []

        i = 0
        return_type = self._parse_single_type(encoding, i)
        i = return_type[1]

        params = []
        while i < len(encoding):
            if encoding[i] == ".":
                params.append("...")
                break
            param_type = self._parse_single_type(encoding, i)
            params.append(param_type[0])
            i = param_type[1]

        return return_type[0], params

    def _parse_single_type(self, encoding: str, start_pos: int) -> Tuple[str, int]:
        i = start_pos
        if i >= len(encoding):
            return "void", i

        prefixes = []
        while i < len(encoding):
            if encoding[i : i + 2] == "LL":
                prefixes.append("long long")
                i += 2
            elif encoding[i] == "L":
                prefixes.append("long")
                i += 1
            elif encoding[i] == "U":
                prefixes.append("unsigned")
                i += 1
            elif encoding[i] == "S":
                prefixes.append("signed")
                i += 1
            elif encoding[i] in "ZWNOI":
                i += 1
            else:
                break

        if i >= len(encoding):
            return "void", i

        base_type = self.base_types.get(encoding[i], f"UnknownType_{encoding[i]}")
        i += 1

        cv = []
        ptrs = []
        while i < len(encoding):
            ch = encoding[i]
            if ch == "*":
                ptrs.append("*")
                i += 1
            elif ch == "&":
                ptrs.append("&")
                i += 1
            elif ch == "C":
                cv.append("const")
                i += 1
            elif ch == "D":
                cv.append("volatile")
                i += 1
            elif ch == "R":
                cv.append("restrict")
                i += 1
            else:
                break

        prefix_str = (" ".join(prefixes) + " ") if prefixes else ""
        cv_str = (" " + " ".join(cv)) if cv else ""
        ptr_str = "".join((" *" if p == "*" else " &") for p in ptrs)
        return f"{prefix_str}{base_type}{cv_str}{ptr_str}".strip(), i

    def decode_attributes(self, attr_str: str) -> List[str]:
        attrs = []
        for char in attr_str:
            if char in self.attributes:
                attrs.append(self.attributes[char])
        return attrs

    def convert_file(self, input_file: str, output_file: str = None):
        try:
            with open(input_file, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: File not found {input_file}")
            return

        builtins = []
        for line in lines:
            line = line.strip()
            if line.startswith("BUILTIN"):
                parsed = self.parse_builtin_line(line)
                if parsed:
                    builtins.append(parsed)

        # Generate output
        output_lines = []

        # Header
        output_lines.append(
            "//===--- BuiltinsXCore.td - XCore Builtin function database ----*- C++ -*-===//"
        )
        output_lines.append("//")
        output_lines.append(
            "// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions."
        )
        output_lines.append(
            "// See https://llvm.org/LICENSE.txt for license information."
        )
        output_lines.append(
            "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception"
        )
        output_lines.append("//")
        output_lines.append(
            "//===----------------------------------------------------------------------===//"
        )
        output_lines.append("")
        output_lines.append('include "clang/Basic/BuiltinsBase.td"')
        output_lines.append("")

        # Base class
        output_lines.append(
            "class XCoreBuiltin<string prototype, list<Attribute> Attr = []> : TargetBuiltin {"
        )
        output_lines.append("  let Spellings = [NAME];")
        output_lines.append("  let Prototype = prototype;")
        output_lines.append("  let Attributes = Attr;")
        output_lines.append("}")
        output_lines.append("")

        # Builtins
        for name, proto_encoding, attr_encoding in builtins:
            try:
                return_type, param_types = self.parse_type_encoding(proto_encoding)
                attributes = self.decode_attributes(attr_encoding)

                # Build prototype
                if not param_types:
                    prototype = f"{return_type}()"
                else:
                    prototype = f"{return_type}({', '.join(param_types)})"

                # Build definition
                if attributes:
                    attr_str = f", [{', '.join(attributes)}]"
                else:
                    attr_str = ""

                output_lines.append(
                    f'def {name} : XCoreBuiltin<"{prototype}"{attr_str}>;'
                )

            except Exception as e:
                output_lines.append(f"// ERROR converting {name}: {e}")

        output_content = "\n".join(output_lines) + "\n"

        if output_file:
            with open(output_file, "w") as f:
                f.write(output_content)
            print(f"Conversion completed!")
            print(f"Output file: {output_file}")
            print(f"Successfully converted: {len(builtins)} functions")
        else:
            print(output_content)


def main():
    converter = XCoreConverter()

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "BuiltinsXCore.td"
        converter.convert_file(input_file, output_file)
    else:
        print("XCore Builtin Function Converter")
        print("Usage:")
        print("  python convert_xcore.py input.def [output.td]")
        print("")
        print("Example:")
        print("  python convert_xcore.py BuiltinsXCore.def BuiltinsXCore.td")


if __name__ == "__main__":
    main()
