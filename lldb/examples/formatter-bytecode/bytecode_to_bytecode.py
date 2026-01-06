#!/usr/bin/python3

import dis
import sys
from types import CodeType
from typing import Iterable, Iterator, cast


# TODO: strlen, fmt
_SELECTORS = {
    "Cast": "@cast",
    "GetChildAtIndex": "@get_child_at_index",
    "GetChildIndex": "@get_child_index",
    "GetChildMemberWithName": "@get_child_with_name",
    "GetNumChildren": "@get_num_children",
    "GetSummary": "@summary",
    "GetTemplateArgumentType": "@get_template_argument_type",
    "GetType": "@get_type",
    "GetValue": "@get_value",
    "GetValueAsAddress": "@get_value_as_address",
    "GetValueAsSigned": "@get_value_as_signed",
    "GetValueAsUnsigned": "@get_value_as_unsigned",
}


def _main(source_file):
    with open(source_file) as f:
        source_code = f.read()
    bytecode = dis.Bytecode(source_code)
    for func_body in _function_bodies(bytecode):
        instructions = dis.get_instructions(func_body)
        for op in _translate(instructions):
            print(op)


def _function_bodies(bytecode: dis.Bytecode) -> Iterable[CodeType]:
    """
    Iterate the function bodies (code object children) of the given Bytecode.
    """
    for const in bytecode.codeobj.co_consts:
        if hasattr(const, "co_code"):
            yield const


def _translate(instructions: Iterator[dis.Instruction]) -> list[str]:
    """
    Convert Python instructions to LLDB data formatter bytecode operations.
    """
    result = []
    _translate_list(list(instructions), result)
    return result


def _translate_list(instructions: list[dis.Instruction], result: list[str]):
    """
    Convert sequences of Python bytecode to sequences of LLDB data formatter
    bytecode.

    This function performs course grained translations - sequences of input to
    sequences of output. For translations of individual instructions, see
    `_translate_instruction`.
    """
    while instructions:
        inst = instructions.pop(0)
        op = inst.opname
        if op == "LOAD_METHOD":
            # Method call sequences begin with a LOAD_METHOD instruction, then
            # load the arguments on to the stack, and end with the CALL_METHOD
            # instruction.
            if selector := _SELECTORS.get(inst.argval):
                while instructions:
                    if instructions[0] == "LOAD_METHOD":
                        # Begin a nested method call.
                        _translate_list(instructions, result)
                    else:
                        # TODO: Can LOAD_METHOD, ..., CALL_METHOD sequences
                        # contain flow control? If so this needs to gather
                        # instructions and call `_translate_list`, instead of
                        # handling each instruction individually.
                        x = instructions.pop(0)
                        if x.opname != "CALL_METHOD":
                            result.append(_translate_instruction(x))
                        else:
                            result.append(f"{selector} call")
                            break
        elif op == "POP_JUMP_IF_FALSE":
            # Convert to an `{ ... } if` sequence.
            result.append("{")
            offset = cast(int, inst.arg)
            idx = _index_of_offset(instructions, offset)
            # Split the condional block prefix from the remaining instructions.
            block = instructions[:idx]
            del instructions[:idx]
            _translate_list(block, result)
            result.append("} if")
        else:
            result.append(_translate_instruction(inst))


def _translate_instruction(inst: dis.Instruction) -> str:
    """
    Convert a single Python bytecode instruction to an LLDB data formatter
    bytecode operation.

    This function performs one-to-one translations. For translations of
    sequences of instructions, see `_translate_list`.
    """
    op = inst.opname
    if op == "COMPARE_OP":
        if inst.argval == "==":
            return "="
    elif op == "LOAD_CONST":
        if isinstance(inst.argval, str):
            # TODO: Handle strings with inner double quotes ("). Alternatively,
            # use `repr()` and allow the bytecode assembly to use single quotes.
            return f'"{inst.argval}"'
        elif isinstance(inst.argval, bool):
            num = int(inst.argval)
            return f"{num}"
        else:
            return inst.argrepr
    elif op == "LOAD_FAST":
        return f"{inst.arg} pick # {inst.argval}"
    elif op == "RETURN_VALUE":
        return "return"
    elif op in ("STORE_FAST", "STORE_NAME"):
        # This is fake. There is no `put` operation (yet?).
        return f"{inst.arg} put # {inst.argval}"
    return op


def _index_of_offset(instructions: list[dis.Instruction], offset) -> int:
    """Find the index of the instruction having the given offset."""
    for i, inst in enumerate(instructions):
        if inst.offset == offset:
            return i
    raise ValueError(f"invalid offset: {offset}")


if __name__ == "__main__":
    _main(sys.argv[1])
