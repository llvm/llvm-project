"""
Specification, compiler, disassembler, and interpreter
for LLDB dataformatter bytecode.

See https://lldb.llvm.org/resources/formatterbytecode.html for more details.
"""

from __future__ import annotations

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
# Compiler.
################################################################################


def compile(assembler: str) -> bytearray:
    """Compile assembler into bytecode"""
    # This is a stack of all in-flight/unterminated blocks.
    bytecode = [bytearray()]

    def emit(byte):
        bytecode[-1].append(byte)

    tokens = list(assembler.split(" "))
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
            s = bytearray()
            done = False
            chrs = tok[1:]
            while not done:
                quoted = False
                for c in chrs:
                    if quoted:
                        s.append(ord(c))  # FIXME
                        quoted = False
                    elif c == "\\":
                        quoted = True
                    elif c == '"':
                        done = True
                        break
                        # FIXME assert this is last in token
                    else:
                        s.append(ord(c))
                if not done:
                    s.append(ord(" "))
                    chrs = tokens.pop()

            emit(op_lit_string)
            emit(len(s))
            bytecode[-1].extend(s)
        else:
            emit(opcode[tok])
    assert len(bytecode) == 1  # unterminated {
    return bytecode[0]


################################################################################
# Disassembler.
################################################################################


def disassemble(bytecode: bytearray) -> (str, int):
    """Disassemble bytecode into (assembler, token starts)"""
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
            s = "'"
            while length:
                s += chr(next_byte())
                length -= 1
            asm += '"' + repr(s)[2:]
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


def interpret(bytecode: bytearray, control: list, data: list, tracing: bool = False):
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
                pass
    return data[-1]


if __name__ == "__main__":
    # Work around the fact that one of the local files is called
    # types.py, which breaks some versions of python.
    import os, sys

    path = os.path.abspath(os.path.dirname(__file__))
    sys.path.remove(path)
    import argparse

    parser = argparse.ArgumentParser(
        description="""
    Compiler, disassembler, and interpreter for LLDB dataformatter bytecode.
    See https://lldb.llvm.org/resources/formatterbytecode.html for more details.
    """
    )
    parser.add_argument(
        "-c", "--compile", type=str, help="compile assembler into bytecode"
    )
    parser.add_argument("-d", "--disassemble", type=str, help="disassemble bytecode")
    parser.add_argument("-t", "--test", action="store_true", help="run unit tests")
    args = parser.parse_args()
    if args.compile:
        print(compile(str(args.compile)).hex())

    if args.disassemble:
        print(disassemble(bytearray.fromhex(str(args.disassemble))))

    ############################################################################
    # Tests.
    ############################################################################
    if args.test:
        import unittest

        class TestCompiler(unittest.TestCase):
            def test(self):
                self.assertEqual(compile("1u dup").hex(), "200101")
                self.assertEqual(compile('"1u dup"').hex(), "2206317520647570")
                self.assertEqual(compile("16 < { dup } if").hex(), "21105210010111")
                self.assertEqual(compile('{ { " } " } }').hex(), "100710052203207d20")

                def roundtrip(asm):
                    self.assertEqual(disassemble(compile(asm))[0], asm)

                roundtrip("1u dup")
                roundtrip('1u dup "1u dup"')
                roundtrip("16 < { dup } if")
                roundtrip('{ { " } " } }')

                self.assertEqual(interpret(compile("1 1 +"), [], []), 2)
                self.assertEqual(interpret(compile("2 1 1 + *"), [], []), 4)
                self.assertEqual(
                    interpret(compile('2 1 > { "yes" } { "no" } ifelse'), [], []), "yes"
                )

                import sys

                sys.argv.pop()
                path = os.path.dirname(__file__)
                sys.path.remove
                unittest.main()
