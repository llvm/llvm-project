"""
Specification, assembler, disassembler, and interpreter
for LLDB dataformatter bytecode.

See https://lldb.llvm.org/resources/formatterbytecode.html for more details.
"""

from __future__ import annotations

# Work around the fact that one of the local files is called
# types.py, which breaks some versions of python.
import os, sys

path = os.path.abspath(os.path.dirname(__file__))
if path in sys.path:
    sys.path.remove(path)

import re
import io
import ast
import enum
import textwrap
from copy import copy
from dataclasses import dataclass
from typing import Any, BinaryIO, Optional, Sequence, TextIO, Tuple, Union, cast

BINARY_VERSION = 1

# Types
type_String = 1
type_Int = 2
type_UInt = 3
type_Object = 4
type_Type = 5

# Opcodes
opcode = dict()


def define_opcode(n, mnemonic, name):
    globals()["op_" + name] = n
    if mnemonic:
        opcode[mnemonic] = n
    opcode[n] = mnemonic


define_opcode(1, "dup", "dup")
define_opcode(2, "drop", "drop")
define_opcode(3, "pick", "pick")
define_opcode(4, "over", "over")
define_opcode(5, "swap", "swap")
define_opcode(6, "rot", "rot")

define_opcode(0x10, "{", "begin")
define_opcode(0x11, "if", "if")
define_opcode(0x12, "ifelse", "ifelse")
define_opcode(0x13, "return", "return")

define_opcode(0x20, None, "lit_uint")
define_opcode(0x21, None, "lit_int")
define_opcode(0x22, None, "lit_string")
define_opcode(0x23, None, "lit_selector")

define_opcode(0x2A, "as_int", "as_int")
define_opcode(0x2B, "as_uint", "as_uint")
define_opcode(0x2C, "is_null", "is_null")

define_opcode(0x30, "+", "plus")
define_opcode(0x31, "-", "minus")
define_opcode(0x32, "*", "mul")
define_opcode(0x33, "/", "div")
define_opcode(0x34, "%", "mod")
define_opcode(0x35, "<<", "shl")
define_opcode(0x36, ">>", "shr")

define_opcode(0x40, "&", "and")
define_opcode(0x41, "|", "or")
define_opcode(0x42, "^", "xor")
define_opcode(0x43, "~", "not")

define_opcode(0x50, "=", "eq")
define_opcode(0x51, "!=", "neq")
define_opcode(0x52, "<", "lt")
define_opcode(0x53, ">", "gt")
define_opcode(0x54, "=<", "le")
define_opcode(0x55, ">=", "ge")

define_opcode(0x60, "call", "call")

# Function signatures
sig_summary = 0
sig_init = 1
sig_get_num_children = 2
sig_get_child_index = 3
sig_get_child_at_index = 4
sig_update = 5

SIGNATURES = {
    "summary": sig_summary,
    "init": sig_init,
    "get_num_children": sig_get_num_children,
    "get_child_index": sig_get_child_index,
    "get_child_at_index": sig_get_child_at_index,
    "update": sig_update,
}

SIGNATURE_NAMES = "|".join(SIGNATURES.keys())
SIGNATURE_IDS = {v: k for k, v in SIGNATURES.items()}

# Selectors
selector = dict()


def define_selector(n, name):
    globals()["sel_" + name] = n
    selector["@" + name] = n
    selector[n] = "@" + name


define_selector(0, "summary")
define_selector(1, "type_summary")

define_selector(0x10, "get_num_children")
define_selector(0x11, "get_child_at_index")
define_selector(0x12, "get_child_with_name")
define_selector(0x13, "get_child_index")
define_selector(0x15, "get_type")
define_selector(0x16, "get_template_argument_type")
define_selector(0x17, "cast")
define_selector(0x18, "get_synthetic_value")
define_selector(0x19, "get_non_synthetic_value")
define_selector(0x20, "get_value")
define_selector(0x21, "get_value_as_unsigned")
define_selector(0x22, "get_value_as_signed")
define_selector(0x23, "get_value_as_address")

define_selector(0x40, "read_memory_byte")
define_selector(0x41, "read_memory_uint32")
define_selector(0x42, "read_memory_int32")
define_selector(0x43, "read_memory_unsigned")
define_selector(0x44, "read_memory_signed")
define_selector(0x45, "read_memory_address")
define_selector(0x46, "read_memory")

define_selector(0x50, "fmt")
define_selector(0x51, "sprintf")
define_selector(0x52, "strlen")


################################################################################
# Assembler.
################################################################################

_SIGNATURE_LABEL = re.compile(f"@(?:{SIGNATURE_NAMES}):$")


def _tokenize(assembler: str) -> list[str]:
    """Convert string of assembly into tokens."""
    # With one exception, tokens are sequences of non-space characters.
    # The one exception is string literals, which may have spaces.

    # To parse strings, which can contain escaped contents, use a "Friedl
    # unrolled loop". The high level of such a regex is:
    #     open normal* ( special normal* )* close
    # which for string literals is:
    string_literal = r'" [^"\\]* (?: \\. [^"\\]* )* "'

    return re.findall(rf"{string_literal} | \S+", assembler, re.VERBOSE)


def _segment_by_signature(input: list[str]) -> list[Tuple[str, list[str]]]:
    """Segment the input tokens along signature labels."""
    segments = []

    # Loop state
    signature = None
    tokens = []

    for token in input:
        if _SIGNATURE_LABEL.match(token):
            if signature:
                segments.append((signature, tokens))
            signature = token[1:-1]  # strip leading @, trailing :
            tokens = []
        else:
            tokens.append(token)

    if signature:
        segments.append((signature, tokens))

    return segments


@dataclass
class BytecodeSection:
    """Abstraction of the data serialized to __lldbformatters sections."""

    type_name: str
    flags: int
    signatures: list[Tuple[str, bytes]]

    def validate(self):
        seen = set()
        for sig, _ in self.signatures:
            if sig in seen:
                raise ValueError(f"duplicate signature: {sig}")
            seen.add(sig)

    def _to_binary(self) -> bytes:
        bin = bytearray()
        bin.extend(_to_uleb(len(self.type_name)))
        bin.extend(bytes(self.type_name, encoding="utf-8"))
        bin.extend(_to_byte(self.flags))
        for sig, bc in self.signatures:
            bin.extend(_to_byte(SIGNATURES[sig]))
            bin.extend(_to_uleb(len(bc)))
            bin.extend(bc)

        return bytes(bin)

    def write_binary(self, output: BinaryIO) -> None:
        self.validate()

        bin = self._to_binary()
        output.write(_to_byte(BINARY_VERSION))
        output.write(_to_uleb(len(bin)))
        output.write(self._to_binary())

    def write_source(self, output: TextIO, language: str) -> None:
        if language == "c":
            self.write_c(output)
        elif language == "swift":
            self.write_swift(output)

    class _CBuilder:
        """Helper class for emitting binary data as a C-string literal."""

        entries: list[Tuple[str, str]]

        def __init__(self) -> None:
            self.entries = []

        def emit_byte(self, x: int, comment: str) -> None:
            self.emit_bytes(_to_byte(x), comment)

        def emit_uleb(self, x: int, comment: str) -> None:
            self.emit_bytes(_to_uleb(x), comment)

        def emit_bytes(self, x: bytes, comment: str) -> None:
            # Construct zero pemited hex values with length two.
            string = "".join(f"\\x{b:02x}" for b in x)
            self.emit_string(string, comment)

        def emit_string(self, string: str, comment: str) -> None:
            self.entries.append((f'"{string}"', comment))

    class _SwiftBuilder:
        """Helper class for emitting binary data as a Swift tuple literal."""

        entries: list[Tuple[bytes, str]]

        def __init__(self) -> None:
            self.entries = []

        def emit_byte(self, x: int, comment: str) -> None:
            self.emit_bytes(_to_byte(x), comment)

        def emit_uleb(self, x: int, comment: str) -> None:
            self.emit_bytes(_to_uleb(x), comment)

        def emit_bytes(self, x: bytes, comment: str) -> None:
            self.entries.append((x, comment))

        def emit_string(self, string: str, comment: str) -> None:
            self.emit_bytes(string.encode(), comment)

        @property
        def type_decl(self):
            total_bytes = sum((len(bs) for bs, _ in self.entries))
            element_list = ", ".join(["UInt8"] * total_bytes)
            return f"({element_list})"

    def _build(self, builder) -> None:
        size = len(self._to_binary())
        builder.emit_byte(BINARY_VERSION, "version")
        builder.emit_uleb(size, "remaining record size")
        builder.emit_uleb(len(self.type_name), "type name size")
        builder.emit_string(self.type_name, "type name")
        builder.emit_byte(self.flags, "flags")
        for sig, bc in self.signatures:
            builder.emit_byte(SIGNATURES[sig], f"sig_{sig}")
            builder.emit_uleb(len(bc), "program size")
            builder.emit_bytes(bc, "program")

    @property
    def _var_name(self):
        var_name = re.sub(r"\W", "_", self.type_name)
        return f"_{var_name}_formatter"

    def write_c(self, output: TextIO) -> None:
        self.validate()

        builder = self._CBuilder()
        self._build(builder)

        print(
            textwrap.dedent(
                """
                #ifdef __APPLE__
                #define FORMATTER_SECTION "__DATA_CONST,__lldbformatters"
                #else
                #define FORMATTER_SECTION ".lldbformatters"
                #endif
                """
            ),
            file=output,
        )
        print(
            "__attribute__((used, section(FORMATTER_SECTION)))",
            file=output,
        )
        print(f"unsigned char {self._var_name}[] =", file=output)
        indent = "    "
        for string, comment in builder.entries:
            print(f"{indent}// {comment}", file=output)
            print(f"{indent}{string}", file=output)
        print(";", file=output)

    def write_swift(self, output: TextIO) -> None:
        self.validate()

        builder = self._SwiftBuilder()
        self._build(builder)

        print(
            textwrap.dedent(
                """\
                #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS) || os(visionOS)
                @section("__DATA_CONST,__lldbformatters")
                #else
                @section(".lldbformatters")
                #endif
                @used"""
            ),
            file=output,
        )
        print(
            f"let {self._var_name}: {builder.type_decl} = (",
            file=output,
        )
        indent = "    "
        for bs, comment in builder.entries:
            print(f"{indent}// {comment}", file=output)
            byte_list = ", ".join(f"0x{b:02x}" for b in bs)
            print(f"{indent}{byte_list},", file=output)
        print(")", file=output)


def assemble_file(type_name: str, input: TextIO) -> BytecodeSection:
    input_tokens = _tokenize(input.read())
    signatures = []
    for sig, tokens in _segment_by_signature(input_tokens):
        if tokens:
            signatures.append((sig, assemble_tokens(tokens)))

    return BytecodeSection(type_name, flags=0, signatures=signatures)


def assemble(assembly: str) -> bytes:
    return assemble_tokens(_tokenize(assembly))


def assemble_tokens(tokens: list[str]) -> bytes:
    """Assemble assembly into bytecode"""
    # This is a stack of all in-flight/unterminated blocks.
    bytecode = [bytearray()]

    def emit(byte):
        bytecode[-1].append(byte)

    tokens.reverse()
    while tokens:
        tok = tokens.pop()
        if tok == "":
            pass
        elif tok == "{":
            bytecode.append(bytearray())
        elif tok == "}":
            block = bytecode.pop()
            emit(op_begin)
            emit(len(block))  # FIXME: uleb
            bytecode[-1].extend(block)
        elif tok[0].isdigit():
            if tok[-1] == "u":
                emit(op_lit_uint)
                emit(int(tok[:-1]))  # FIXME
            else:
                emit(op_lit_int)
                emit(int(tok))  # FIXME
        elif tok[0] == "@":
            emit(op_lit_selector)
            emit(selector[tok])
        elif tok[0] == '"':
            # Remove backslash escaping '"' and '\'.
            s = re.sub(r'\\(["\\])', r"\1", tok[1:-1]).encode()
            emit(op_lit_string)
            emit(len(s))
            bytecode[-1].extend(s)
        else:
            emit(opcode[tok])
    assert len(bytecode) == 1  # unterminated {
    return bytes(bytecode[0])


################################################################################
# Disassembler.
################################################################################


def disassemble_file(input: BinaryIO, output: TextIO) -> None:
    stream = io.BytesIO(input.read())

    version = stream.read(1)[0]
    if version != BINARY_VERSION:
        raise ValueError(f"unknown binary version: {version}")

    record_size = _from_uleb(stream)
    stream.truncate(stream.tell() + record_size)

    name_size = _from_uleb(stream)
    _type_name = stream.read(name_size).decode()
    _flags = stream.read(1)[0]

    while True:
        sig_byte = stream.read(1)
        if not sig_byte:
            break
        sig_name = SIGNATURE_IDS[sig_byte[0]]
        body_size = _from_uleb(stream)
        bc = stream.read(body_size)
        asm, _ = disassemble(bc)
        print(f"@{sig_name}: {asm}", file=output)


def disassemble(bytecode: bytes) -> Tuple[str, list[int]]:
    """Disassemble bytecode into (assembly, token starts)"""
    asm = ""
    all_bytes = list(bytecode)
    all_bytes.reverse()
    blocks = []
    tokens = [0]

    def next_byte():
        """Fetch the next byte in the bytecode and keep track of all
        in-flight blocks"""
        for i in range(len(blocks)):
            blocks[i] -= 1
        tokens.append(len(asm))
        return all_bytes.pop()

    while all_bytes:
        b = next_byte()
        if b == op_begin:
            asm += "{"
            length = next_byte()
            blocks.append(length)
        elif b == op_lit_uint:
            b = next_byte()
            asm += str(b)  # FIXME uleb
            asm += "u"
        elif b == op_lit_int:
            b = next_byte()
            asm += str(b)
        elif b == op_lit_selector:
            b = next_byte()
            asm += selector[b]
        elif b == op_lit_string:
            length = next_byte()
            s = '"'
            for _ in range(length):
                c = chr(next_byte())
                if c in ('"', "\\"):
                    s += "\\"
                s += c
            s += '"'
            asm += s
        else:
            asm += opcode[b]

        while blocks and blocks[-1] == 0:
            asm += " }"
            blocks.pop()

        if all_bytes:
            asm += " "

    if blocks:
        asm += "ERROR"
    return asm, tokens


################################################################################
# Interpreter.
################################################################################


def count_fmt_params(fmt: str) -> int:
    """Count the number of parameters in a format string"""
    from string import Formatter

    f = Formatter()
    n = 0
    for _, name, _, _ in f.parse(fmt):
        if name > n:
            n = name
    return n


def interpret(bytecode: bytes, control: list, data: list, tracing: bool = False):
    """Interpret bytecode"""
    frame = []
    frame.append((0, len(bytecode)))

    def trace():
        """print a trace of the execution for debugging purposes"""

        def fmt(d):
            if isinstance(d, int):
                return str(d)
            if isinstance(d, str):
                return d
            return repr(type(d))

        pc, end = frame[-1]
        asm, tokens = disassemble(bytecode)
        print(
            "=== frame = {1}, data = {2}, opcode = {0}".format(
                opcode[b], frame, [fmt(d) for d in data]
            )
        )
        print(asm)
        print(" " * (tokens[pc]) + "^")

    def next_byte():
        """Fetch the next byte and update the PC"""
        pc, end = frame[-1]
        assert pc < len(bytecode)
        b = bytecode[pc]
        frame[-1] = pc + 1, end
        # At the end of a block?
        while pc >= end:
            frame.pop()
            if not frame:
                return None
            pc, end = frame[-1]
            if pc >= end:
                return None
            b = bytecode[pc]
            frame[-1] = pc + 1, end
        return b

    while frame[-1][0] < len(bytecode):
        b = next_byte()
        if b == None:
            break
        if tracing:
            trace()
        # Data stack manipulation.
        if b == op_dup:
            data.append(data[-1])
        elif b == op_drop:
            data.pop()
        elif b == op_pick:
            data.append(data[data.pop()])
        elif b == op_over:
            data.append(data[-2])
        elif b == op_swap:
            x = data.pop()
            y = data.pop()
            data.append(x)
            data.append(y)
        elif b == op_rot:
            z = data.pop()
            y = data.pop()
            x = data.pop()
            data.append(z)
            data.append(x)
            data.append(y)

        # Control stack manipulation.
        elif b == op_begin:
            length = next_byte()
            pc, end = frame[-1]
            control.append((pc, pc + length))
            frame[-1] = pc + length, end
        elif b == op_if:
            if data.pop():
                frame.append(control.pop())
        elif b == op_ifelse:
            if data.pop():
                control.pop()
                frame.append(control.pop())
            else:
                frame.append(control.pop())
                control.pop()
        elif b == op_return:
            control.clear()
            return data[-1]

        # Literals.
        elif b == op_lit_uint:
            b = next_byte()  # FIXME uleb
            data.append(int(b))
        elif b == op_lit_int:
            b = next_byte()  # FIXME uleb
            data.append(int(b))
        elif b == op_lit_selector:
            b = next_byte()
            data.append(b)
        elif b == op_lit_string:
            length = next_byte()
            s = ""
            while length:
                s += chr(next_byte())
                length -= 1
            data.append(s)

        elif b == op_as_uint:
            pass
        elif b == op_as_int:
            pass
        elif b == op_is_null:
            data.append(1 if data.pop() == None else 0)

        # Arithmetic, logic, etc.
        elif b == op_plus:
            data.append(data.pop() + data.pop())
        elif b == op_minus:
            data.append(-data.pop() + data.pop())
        elif b == op_mul:
            data.append(data.pop() * data.pop())
        elif b == op_div:
            y = data.pop()
            data.append(data.pop() / y)
        elif b == op_mod:
            y = data.pop()
            data.append(data.pop() % y)
        elif b == op_shl:
            y = data.pop()
            data.append(data.pop() << y)
        elif b == op_shr:
            y = data.pop()
            data.append(data.pop() >> y)
        elif b == op_and:
            data.append(data.pop() & data.pop())
        elif b == op_or:
            data.append(data.pop() | data.pop())
        elif b == op_xor:
            data.append(data.pop() ^ data.pop())
        elif b == op_not:
            data.append(not data.pop())
        elif b == op_eq:
            data.append(data.pop() == data.pop())
        elif b == op_neq:
            data.append(data.pop() != data.pop())
        elif b == op_lt:
            data.append(data.pop() > data.pop())
        elif b == op_gt:
            data.append(data.pop() < data.pop())
        elif b == op_le:
            data.append(data.pop() >= data.pop())
        elif b == op_ge:
            data.append(data.pop() <= data.pop())

        # Function calls.
        elif b == op_call:
            sel = data.pop()
            if sel == sel_summary:
                data.append(data.pop().GetSummary())
            elif sel == sel_get_num_children:
                data.append(data.pop().GetNumChildren())
            elif sel == sel_get_child_at_index:
                index = data.pop()
                valobj = data.pop()
                data.append(valobj.GetChildAtIndex(index))
            elif sel == sel_get_child_with_name:
                name = data.pop()
                valobj = data.pop()
                data.append(valobj.GetChildMemberWithName(name))
            elif sel == sel_get_child_index:
                name = data.pop()
                valobj = data.pop()
                data.append(valobj.GetIndexOfChildWithName(name))
            elif sel == sel_get_type:
                data.append(data.pop().GetType())
            elif sel == sel_get_template_argument_type:
                n = data.pop()
                valobj = data.pop()
                data.append(valobj.GetTemplateArgumentType(n))
            elif sel == sel_get_synthetic_value:
                data.append(data.pop().GetSyntheticValue())
            elif sel == sel_get_non_synthetic_value:
                data.append(data.pop().GetNonSyntheticValue())
            elif sel == sel_get_value:
                data.append(data.pop().GetValue())
            elif sel == sel_get_value_as_unsigned:
                data.append(data.pop().GetValueAsUnsigned())
            elif sel == sel_get_value_as_signed:
                data.append(data.pop().GetValueAsSigned())
            elif sel == sel_get_value_as_address:
                data.append(data.pop().GetValueAsAddress())
            elif sel == sel_cast:
                sbtype = data.pop()
                valobj = data.pop()
                data.append(valobj.Cast(sbtype))
            elif sel == sel_strlen:
                s = data.pop()
                data.append(len(s) if s else 0)
            elif sel == sel_fmt:
                fmt = data.pop()
                n = count_fmt_params(fmt)
                args = []
                for i in range(n):
                    args.append(data.pop())
                data.append(fmt.format(*args))
            else:
                print("not implemented: " + selector[sel])
                assert False
    return data[-1]


################################################################################
# Python -> Bytecode Compiler
################################################################################

_BUILTINS = {
    "Cast": "@cast",
    "GetChildAtIndex": "@get_child_at_index",
    "GetChildMemberWithName": "@get_child_with_name",
    "GetSummary": "@summary",
    "GetSyntheticValue": "@get_synthetic_value",
    "GetTemplateArgumentType": "@get_template_argument_type",
    "GetType": "@get_type",
    "GetValueAsUnsigned": "@get_value_as_unsigned",
}

_COMPS = {
    ast.Eq: "=",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "=<",
    ast.Gt: ">",
    ast.GtE: "=>",
}

# Maps Python method names in a formatter class to their bytecode signatures.
_METHOD_SIGS = {
    "__init__": "@init",
    "update": "@update",
    "num_children": "@get_num_children",
    "get_child_index": "@get_child_index",
    "get_child_at_index": "@get_child_at_index",
    "get_value": "@get_value",
}


class CompilerError(Exception):
    lineno: int

    def __init__(self, message, node: Union[ast.expr, ast.stmt]) -> None:
        super().__init__(message)
        self.lineno = node.lineno


class Compiler(ast.NodeVisitor):
    """
    Compile Python LLDB data formatters to LLDB formatter bytecode.

    This compiler is supports a limited subset of Python.

    # Supported Features

    * Top level functions implementing LLDB summary formatters
    * Top level classes implementing LLDB synthetic formatters
    * Partial support for the following, see below for more details:
      - Object attributes (properties)
      - Local variables
      - Function calls
    * Python language support
    [x] If statements (including else, elif and nested if)
    [x] Return statements
    [x] String, integer, float, boolean, and None literals
    [x] Binary comparisons
    [ ] Boolean operators
    [ ] Math operations

    # Unsupported Features

        Note: that this is not exhaustive, refer to the list of supported
        features above.

    * For and while loops
    * Exceptions
    * User defined general purpose functions and classes
    * Lists, dicts, sets, and other container data types
    * Iterators, comprehensions, yield, etc
    * With statements
    * Imports of any modules

    # Variables

    The compiler supports two kinds of variables, local variables and attribute
    variables (properties), but there are limitations on both.

    In __init__ and update, local variables are currently *not* supported, but
    attributes can be assigned to. This matches the common case for these
    functions.

    In all other function bodies, local variables _are_ supported, but
    attributes can only be read from, *not* assigned to. This also matches the
    common case for these functions.

    Variables (local and attributes) are tracked, allowing the compiler to know
    their position in the stack. Variable reads can then be lowered to `pick`
    instructions. See the compiler's `locals` and `attrs` attributes.

    # Functions

    Known functions are supported, a design that customizes the scope of what
    formatters can and can't do. The functions known to the compiler are called
    "selectors". The selectors are primarily SBValue API, although there are
    also general purpose selectors. Formatters can only call selectors, not user
    defined functions, and not SB methods that have not been defined as a
    selector.
    """

    # Names of locals in bottom-to-top stack order. locals[0] is the
    # oldest/deepest; locals[-1] is the most recently pushed.
    locals: list[str]

    # Names of visible attrs in bottom-to-top stack order. Always holds the
    # full combined frame for the method being compiled: grows incrementally
    # during __init__/update, and is set to the combined list before getter
    # methods are compiled.
    attrs: list[str]

    # Temporaries currently on the stack above the locals/attrs frame.
    # Always 0 at statement boundaries.
    num_temps: int

    # Bytecode signature of the method being compiled, or None for top-level
    # functions.
    current_sig: Optional[str]

    buffer: io.StringIO

    def __init__(self) -> None:
        self.locals = []
        self.attrs = []
        self.num_temps = 0
        self.current_sig = None
        self.buffer = io.StringIO()

    def compile(self, source_file: str) -> str:
        with open(source_file) as f:
            root = ast.parse(f.read())
        self.visit(root)
        return self.buffer.getvalue()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Compile methods in a fixed order so that attrs is fully populated
        # before getter methods are compiled.
        methods = {}
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name not in _METHOD_SIGS:
                    raise CompilerError(f"unsupported method: {item.name}", item)
                methods[item.name] = item

        self.attrs = []
        if method := methods.get("__init__"):
            self._compile_method(method)
        # self.attrs now holds init's attrs. update's attrs are appended above
        # them, so after update self.attrs is the combined init+update list.
        if method := methods.get("update"):
            self._compile_method(method)

        for method_name, method in methods.items():
            if method_name not in ("__init__", "update"):
                self._compile_method(method)

    def _compile_method(self, node: ast.FunctionDef) -> None:
        self.current_sig = _METHOD_SIGS[node.name]
        self.num_temps = 0

        # Strip 'self' (and 'internal_dict' for __init__) from the arg list;
        # the remaining args become the initial locals.
        args = copy(node.args.args)
        args.pop(0)  # drop 'self'
        if node.name == "__init__":
            args.pop()  # drop trailing 'internal_dict'

        self.locals = [arg.arg for arg in args]

        # Compile into a temporary buffer so the signature line can be
        # emitted first.
        saved_buffer = self.buffer
        self.buffer = io.StringIO()

        self._visit_each(node.body)

        method_output = self.buffer.getvalue()
        self.buffer = saved_buffer
        self._output(f"{self.current_sig}:")
        self._output(method_output)

        self.locals.clear()
        self.current_sig = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Top-level function (not inside a class).
        self.current_sig = None
        self.attrs = []
        self.locals = [arg.arg for arg in node.args.args]
        self._visit_each(node.body)
        self.locals.clear()

    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        # XXX: Does not handle multiple comparisons, ex: `0 < x < 10`
        self.visit(node.comparators[0])
        self._output(_COMPS[type(node.ops[0])])
        # The comparison consumes two values and produces one.
        self.num_temps -= 1

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        # `if`/`ifelse` consumes the condition.
        self.num_temps = 0

        self._output("{")
        self._visit_each(node.body)
        if node.orelse:
            self.num_temps = 0
            self._output("} {")
            self._visit_each(node.orelse)
            self._output("} ifelse")
        else:
            self._output("} if")

    def visit_Return(self, node: ast.Return) -> None:
        self.num_temps = 0
        if node.value:
            self.visit(node.value)
        self._output("return")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._output(f'"{node.value}"')
        elif isinstance(node.value, bool):
            self._output(int(node.value))
        else:
            self._output(node.value)
        self.num_temps += 1

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            receiver = func.value
            method = func.attr
            # self is not a valid call receiver.
            if isinstance(receiver, ast.Name) and receiver.id == "self":
                raise CompilerError(
                    "self is not a valid call receiver; use self.attr to read an attribute",
                    node,
                )
            if selector := _BUILTINS.get(method):
                self.visit(receiver)
                self._visit_each(node.args)
                self._output(f"{selector} call")
                # `call` pops the receiver and all args, and pushes one result.
                self.num_temps -= len(node.args)
                return
            raise CompilerError(f"unsupported method: {method}", node)

        if isinstance(func, ast.Name):
            raise CompilerError(f"unsupported function: {func.id}", node)

        raise CompilerError("unsupported function call expression", node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.num_temps = 0

        target = node.targets[0]

        # Handle self.attr = expr (attribute assignment).
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            if self.current_sig not in ("@init", "@update"):
                raise CompilerError(
                    "attribute assignment is only allowed in __init__ and update",
                    node,
                )

            attr = target.attr
            if attr in self.attrs:
                raise CompilerError(f"attribute '{attr}' is already assigned", node)

            # If the RHS is an argument (the only kind of local permitted in
            # __init__) - then it is already on the stack in place, and no
            # evaluation is needed.
            is_arg = (
                isinstance(node.value, ast.Name)
                and self._local_index(node.value) is not None
            )
            if not is_arg:
                # Evaluate the RHS, leaving its value on the stack.
                self.visit(node.value)

            # Record the attr.
            self.attrs.append(attr)
            return

        # Handle local variable assignment.
        if self.current_sig in ("@init", "@update"):
            raise CompilerError(
                "local variable assignment is not allowed in __init__ or update; "
                "use attribute assignment (self.attr = ...) instead",
                node,
            )

        if isinstance(target, ast.Name):
            names = [target]
        elif isinstance(target, ast.Tuple):
            names = cast(list[ast.Name], target.elts)
        else:
            raise CompilerError("unsupported assignment target", node)

        # Visit RHS, leaving its value on the stack.
        self.visit(node.value)

        # Forget any previous bindings of these names.
        # Their values are orphaned on the stack.
        for name in names:
            idx = self._local_index(name)
            if idx is not None:
                self.locals[idx] = ""

        self.locals.extend(x.id for x in names)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Only self.attr reads are supported here.
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            raise CompilerError(
                "unsupported attribute access (only self.attr is supported)", node
            )
        attr_idx = self._attr_index(node.attr, node)
        pick_idx = self.num_temps + attr_idx
        self._output(f"{pick_idx}u pick")  # "# self.{node.attr}"
        self.num_temps += 1

    def visit_Name(self, node: ast.Name) -> None:
        idx = self._stack_index(node)
        if idx is None:
            raise CompilerError(f"unknown local variable: {node.id}", node)
        self._output(f"{idx}u pick")  # "# {node.id}"
        self.num_temps += 1

    def _visit_each(self, nodes: Sequence[ast.AST]) -> None:
        for child in nodes:
            self.visit(child)

    def _attr_index(self, name: str, node: ast.expr) -> int:
        # self.attrs is always the full visible attr frame, so the index is
        # the direct pick offset with no further adjustment.
        try:
            return self.attrs.index(name)
        except ValueError:
            raise CompilerError(f"unknown attribute: {name}", node)

    def _stack_index(self, name: ast.Name) -> Optional[int]:
        # Offset past all attrs and any in-flight temporaries.
        idx = self._local_index(name)
        if idx is None:
            return None
        return len(self.attrs) + idx + self.num_temps

    def _local_index(self, name: ast.Name) -> Optional[int]:
        try:
            return self.locals.index(name.id)
        except ValueError:
            return None

    def _output(self, x: Any) -> None:
        print(x, file=self.buffer)


################################################################################
# Helper functions.
################################################################################


def _to_uleb(value: int) -> bytes:
    """Encode an integer to ULEB128 bytes."""
    if value < 0:
        raise ValueError(f"negative number cannot be encoded to ULEB128: {value}")

    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        result.append(byte)
        if value == 0:
            break

    return bytes(result)


def _from_uleb(stream: BinaryIO) -> int:
    """Decode a ULEB128 integer by reading bytes from the stream."""
    result = 0
    shift = 0
    while True:
        byte = stream.read(1)[0]
        result |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            break

    return result


def _to_byte(n: int) -> bytes:
    return n.to_bytes(1, "big")


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""
    Assembler, disassembler, and interpreter for LLDB dataformatter bytecode.
    See https://lldb.llvm.org/resources/formatterbytecode.html for more details.
    """
    )
    parser.add_argument("input", help="input file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="compile a Python LLDB data formatter into LLDB formatter bytecode",
    )
    mode.add_argument(
        "-a",
        "--assemble",
        action="store_true",
        help="assemble assembly into bytecode",
    )
    mode.add_argument(
        "-d",
        "--disassemble",
        action="store_true",
        help="disassemble bytecode",
    )
    parser.add_argument("-n", "--type-name", help="source type of formatter")
    parser.add_argument(
        "-o",
        "--output",
        help="output file (required for --assemble)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("binary", "c", "swift"),
        default="binary",
        help="output file format",
    )
    parser.add_argument("-t", "--test", action="store_true", help="run unit tests")

    args = parser.parse_args()
    if args.compile:
        if not args.type_name:
            parser.error("--type-name is required with --compile")
        if not args.output:
            parser.error("--output is required with --compile")
        compiler = Compiler()
        try:
            assembly = compiler.compile(args.input)
        except CompilerError as e:
            print(f"{args.input}:{e.lineno}: {e}", file=sys.stderr)
            return

        section = assemble_file(args.type_name, io.StringIO(assembly))
        if args.format == "binary":
            with open(args.output, "wb") as output:
                section.write_binary(output)
        else:
            with open(args.output, "w") as output:
                section.write_source(output, language=args.format)
    elif args.assemble:
        if not args.type_name:
            parser.error("--type-name is required with --assemble")
        if not args.output:
            parser.error("--output is required with --assemble")
        with open(args.input) as input:
            section = assemble_file(args.type_name, input)
        if args.format == "binary":
            with open(args.output, "wb") as output:
                section.write_binary(output)
        else:
            with open(args.output, "w") as output:
                section.write_source(output, language=args.format)
    elif args.disassemble:
        if args.output:
            with (
                open(args.input, "rb") as input,
                open(args.output, "w") as output,
            ):
                disassemble_file(input, output)
        else:
            with open(args.input, "rb") as input:
                disassemble_file(input, sys.stdout)


if __name__ == "__main__":
    if not ("-t" in sys.argv or "--test" in sys.argv):
        _main()
        sys.exit()

    ############################################################################
    # Tests.
    ############################################################################
    import unittest

    class TestAssembler(unittest.TestCase):

        def test_assemble(self):
            self.assertEqual(assemble("1u dup").hex(), "200101")
            self.assertEqual(assemble('"1u dup"').hex(), "2206317520647570")
            self.assertEqual(assemble("16 < { dup } if").hex(), "21105210010111")
            self.assertEqual(assemble('{ { " } " } }').hex(), "100710052203207d20")

            def roundtrip(asm):
                self.assertEqual(disassemble(assemble(asm))[0], asm)

            roundtrip("1u dup")
            roundtrip("16 < { dup } if")
            roundtrip('{ { " } " } }')

            # String specific checks.
            roundtrip('1u "2u 3u"')
            roundtrip('"a  b"')
            roundtrip('"a \\" b"')

            self.assertEqual(interpret(assemble("1 1 +"), [], []), 2)
            self.assertEqual(interpret(assemble("2 1 1 + *"), [], []), 4)
            self.assertEqual(
                interpret(assemble('2 1 > { "yes" } { "no" } ifelse'), [], []), "yes"
            )

        def test_assemble_file(self):
            def run_assemble(type_name, asm):
                out = io.BytesIO()
                section = assemble_file(type_name, io.StringIO(asm))
                section.write_binary(out)
                out.seek(0)
                return out

            def run_disassemble(binary):
                out = io.StringIO()
                disassemble_file(binary, out)
                out.seek(0)
                return out

            # assemble -> disassemble -> assemble round-trip: binary is identical.
            asm = "@summary: dup @get_value_as_unsigned call return\n@get_num_children: drop 5u return"
            binary1 = run_assemble("MyType", asm)
            dis = run_disassemble(binary1)
            binary2 = run_assemble("MyType", dis.read())
            self.assertEqual(binary1.getvalue(), binary2.getvalue())

            # disassemble -> assemble -> disassemble round-trip: text is identical.
            dis2 = run_disassemble(binary2)
            self.assertEqual(dis.getvalue(), dis2.getvalue())

            # disassemble output contains expected signatures.
            self.assertIn("@summary:", dis.getvalue())
            self.assertIn("@get_num_children:", dis.getvalue())

            # Duplicate signature is an error.
            with self.assertRaises(ValueError):
                run_assemble("MyType", "@summary: 1u return\n@summary: 2u return")

        def test_write_source(self):
            # Use the Account example from main.cpp as a reference, whose
            # exact byte values are known.
            section = BytecodeSection(
                type_name="Account",
                flags=0,
                signatures=[
                    ("get_num_children", bytes([0x20, 0x01])),
                    ("get_child_at_index", bytes([0x02, 0x20, 0x00, 0x23, 0x11, 0x60])),
                ],
            )
            out = io.StringIO()
            section.write_source(out, language="c")
            src = out.getvalue()

            self.assertIn("__attribute__((used, section(FORMATTER_SECTION)))", src)
            self.assertIn("unsigned char _Account_formatter[] =", src)
            self.assertIn('"\\x01"', src)  # version
            self.assertIn('"\\x15"', src)  # record size (21)
            self.assertIn('"\\x07"', src)  # type name size (7)
            self.assertIn('"Account"', src)  # type name
            self.assertIn('"\\x00"', src)  # flags
            self.assertIn('"\\x02"', src)  # sig_get_num_children
            self.assertIn('"\\x20\\x01"', src)  # program
            self.assertIn('"\\x04"', src)  # sig_get_child_at_index
            self.assertIn('"\\x06"', src)  # program size
            self.assertIn('"\\x02\\x20\\x00\\x23\\x11\\x60"', src)  # program
            self.assertIn("// version", src)
            self.assertIn("// type name", src)
            self.assertIn("// program", src)
            # Semicolon terminates the array initializer.
            self.assertEqual(src.count(";"), 1)

            # Non-identifier characters in the type name are replaced with '_'.
            out2 = io.StringIO()
            BytecodeSection("std::vector<int>", 0, []).write_source(out2, language="c")
            self.assertIn("_std__vector_int__formatter[] =", out2.getvalue())

    unittest.main(argv=[__file__])
