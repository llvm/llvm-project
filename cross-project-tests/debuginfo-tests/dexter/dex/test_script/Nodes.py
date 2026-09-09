# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""This file defines all of the Nodes used to create Dexter scripts. All Nodes must be registered with the yaml
constructor/representer in `setup_yaml_parser` before loading or printing any script.
"""

import abc
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Union
import yaml
from dex.dextIR.ValueIR import ValueIR
from dex.utils.Exceptions import Error


def setup_yaml_parser(loader):
    reg_classes = [
        Address,
        DexRange,
        Float,
        Label,
        Step,
        Then,
        Type,
        TypeAll,
        Value,
        ValueAll,
        Where,
    ]
    for c in reg_classes:
        c.register_yaml(loader)


class DexterNodeError(Error):
    """Class representing errors with Dexter node parsing."""

    def __init__(self, node, msg):
        super(DexterNodeError, self).__init__(msg)
        self.msg = msg
        self.node = node

    def __str__(self):
        return f"Error with node: {self.node}: {self.msg}"


@dataclass
class FileLabels:
    """Small utility class for passing the set of labels associated with a particular file."""

    file: str
    labels: Dict[str, int]


###################
## Structural Nodes: These are used as keys in the Script, and collectively define Dexter's actions when running a test:
##                   how it steps and navigates through the debuggee program, and what information it collects from the
##                   debugger.


class Where:
    """One or more instances of this class define a range of steps in a debugging session. Any expects in the script
    within scope of a "Where" will only be evaluated for the steps where the Where applies.

    Supports a set of attributes to specify the state of a stack frame that this node matches against:
    - file: The path (absolute or relative) of the frame source location.
    - line: The line number of the frame source location.
    - function: The function name (exactly matching the name shown by the debugger) of the frame.
    - at_frame_idx: Only specifiable for !and nodes; changes the frame that this node matches against from its parent
                    frame to the frame at the specified index.
    - after_hit_count: Requires that this node must become "active" a specified number of times before it will be
                       considered as active. If for_hit_count is also specified, the "for" hit count will only begin
                       accumulating after the "after" hit count.
    - for_hit_count: Limits this node to becoming active a specified number of times, after which it will not become
                     active again. If after_hit_count is also specified, the "for" hit count will only begin
                     accumulating after the "after" hit count.
    """

    def __init__(self, attributes: dict, is_and: bool):
        self.file: Optional[str] = attributes.pop("file", None)
        self.function: Union[list[str], str, None] = attributes.pop("function", None)
        lines = attributes.pop("lines", None)
        if isinstance(lines, (int, Label)):
            lines = Line(lines)
        self.lines: Union[Line, DexRange, None] = lines
        self.at_frame_idx: Optional[int] = attributes.pop("at_frame_idx", None)
        self.after_hit_count: Optional[int] = attributes.pop("after_hit_count", None)
        self.for_hit_count: Optional[int] = attributes.pop("for_hit_count", None)
        self.conditions: Optional[str] = attributes.pop("conditions", None)
        self.is_and = is_and
        if attributes:
            raise DexterNodeError(
                self, f"unexpected attributes {', '.join(attributes)}"
            )
        if self.at_frame_idx is not None and not self.is_and:
            raise DexterNodeError(self, "at_frame_idx can only be used with !and nodes")

    def __repr__(self):
        elts = [
            f"{name}={value}"
            for name, value in self.get_attrs().items()
            if value is not None
        ]
        name = "And" if self.is_and else "Where"
        return f"{name}(" + ", ".join(elts) + ")"

    def get_attrs(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "function": self.function,
            "lines": self.lines.value if isinstance(self.lines, Line) else self.lines,
            "at_frame_idx": self.at_frame_idx,
            "for_hit_count": self.for_hit_count,
            "after_hit_count": self.after_hit_count,
            "conditions": self.conditions,
        }

    @staticmethod
    def get_constructor(is_and: bool):
        def constructor(loader, node):
            return Where(loader.construct_mapping(node), is_and)

        return constructor

    @staticmethod
    def representer(dumper: yaml.Dumper, data: "Where"):
        mapping = {
            name: value for name, value in data.get_attrs().items() if value is not None
        }
        tag = "!and" if data.is_and else "!where"
        return dumper.represent_mapping(tag, mapping, flow_style=True)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!where", Where.get_constructor(False), loader)
        yaml.add_constructor("!and", Where.get_constructor(True), loader)
        yaml.add_representer(Where, Where.representer)

    def get_lines(self, labels: FileLabels) -> range:
        """Returns the range of line numbers that this Where references, returning an empty range if this Where does not
        refer to any lines."""
        if not self.lines:
            return range(-1)
        if isinstance(self.lines, Line):
            line_num = self.lines.to_line(labels)
            return range(line_num, line_num + 1)
        assert isinstance(
            self.lines, DexRange
        ), f"Invalid type for lines: {self.lines}: ({type(self.lines)})"
        return self.lines.to_range(labels)


###################
## Expect Nodes: These nodes define the expected outputs from the debugger - they are the only nodes that produce
##               metrics, and map to an expected value in the script.


class Expect:
    """An expectation of some debugger state that will be compared to actual observed debugger state and generate one
    or more metrics as a measurement of the difference.
    Expects are largely evaluated independently, but may influence each other through the evaluation context.
    """

    @staticmethod
    def get_variable_result(value: ValueIR) -> Optional[str]:
        """For Expects that extract actual results from ValueIR, this method returns that result from the given value,
        excluding any subvalues (i.e. struct members), or None if there is no valid result for this ValueIR.
        """

    def get_watched_expr(self) -> Optional[str]:
        """Returns the expression that this Expect wants to evaluate."""
        return None

    def get_watched_scope(self) -> Optional[str]:
        """Returns the scope that this Expect wants to evaluate."""
        return None


class ExpectAll(Expect):
    """An Expect for all variables within a named debugger scope; used only to generate scripts from debugger output,
    cannot be used for testing debugger output directly.
    """

    @staticmethod
    def get_base_expect(var_name: str) -> Expect:
        raise NotImplementedError(f"No ExpectAll base type declared")


class Value(Expect):
    """Expect node used to test the value(s) for a single variable, functioning similarly to !value. Allows expecting
    a single value, a list of values, and/or values of members for aggregate variables.

    This node compares the expected values against the actual values observed in the debugger, and produces a set of
    metrics quantifying the difference between expected and actual. Can be used with script rewriting to generate
    expected values for the tested variable.
    """

    def __init__(self, variable_name: str):
        self.variable_name = variable_name
        self.actual_values = None

    @staticmethod
    def get_variable_result(value: ValueIR) -> Optional[str]:
        if value.could_evaluate and not (
            value.is_irretrievable or value.is_optimized_away
        ):
            return value.value
        return None

    def get_watched_expr(self) -> str:
        return self.variable_name

    def __repr__(self):
        return f"Value({self.variable_name})"

    @staticmethod
    def constructor(loader: yaml.Loader, node):
        return Value(loader.construct_scalar(node))

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar("!value", data.variable_name)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!value", Value.constructor, loader)
        yaml.add_representer(Value, Value.representer)


class ValueAll(ExpectAll):
    """Expect node used to write values for all variables within a particular debugger scope, as defined by the DAP
    specification; see: https://microsoft.github.io/debug-adapter-protocol/specification#Requests_Scopes.

    This node is not directly evaluated; it must have no expected values, and when Dexter rewrites the original script,
    this node will be replaced with !value nodes for each variable that was seen in its scope inserted under !and nodes
    that cover that variable's live range(s).
    """

    def __init__(self, scope_name: str):
        self.scope_name = scope_name

    def __repr__(self):
        return f"ValueAll({self.scope_name})"

    @staticmethod
    def get_base_expect(var_name: str) -> Expect:
        return Value(var_name)

    @staticmethod
    def get_variable_result(value: ValueIR) -> Optional[str]:
        return Value.get_variable_result(value)

    def get_watched_scope(self) -> Optional[str]:
        return self.scope_name

    @staticmethod
    def constructor(loader, node):
        return ValueAll(loader.construct_scalar(node))

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar("!value/all", data.scope_name)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!value/all", ValueAll.constructor, loader)
        yaml.add_representer(ValueAll, ValueAll.representer)


class Type(Expect):
    """Expect node used to test the type(s) for a single variable, functioning similarly to !value. Allows expecting
    a single type, a list of types, and/or types of members for aggregate variables.

    This node compares the expected types against the actual types observed in the debugger, and produces a set of
    metrics quantifying the difference between expected and actual. Can be used with script rewriting to generate
    expected types for the tested variable.
    """

    def __init__(self, variable_name: str):
        self.variable_name = variable_name
        self.actual_values = None

    @staticmethod
    def get_variable_result(value: ValueIR) -> Optional[str]:
        if value.could_evaluate:
            return value.type_name
        return None

    def get_watched_expr(self) -> str:
        return self.variable_name

    def __repr__(self):
        return f"Type({self.variable_name})"

    @staticmethod
    def constructor(loader: yaml.Loader, node):
        return Type(loader.construct_scalar(node))

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar("!type", data.variable_name)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!type", Type.constructor, loader)
        yaml.add_representer(Type, Type.representer)


class TypeAll(ExpectAll):
    """Expect node used to write types for all variables within a particular debugger scope, as defined by the DAP
    specification; see: https://microsoft.github.io/debug-adapter-protocol/specification#Requests_Scopes.

    This node is not directly evaluated; it must have no expected values, and when Dexter rewrites the original script,
    this node will be replaced with !type nodes for each variable that was seen in its scope inserted under !and nodes
    that cover that variable's live range(s).
    """

    def __init__(self, scope_name: str):
        self.scope_name = scope_name

    def __repr__(self):
        return f"TypeAll({self.scope_name})"

    @staticmethod
    def get_base_expect(var_name: str) -> Expect:
        return Type(var_name)

    @staticmethod
    def get_variable_result(value: ValueIR) -> Optional[str]:
        return Type.get_variable_result(value)

    def get_watched_scope(self) -> Optional[str]:
        return self.scope_name

    @staticmethod
    def constructor(loader, node):
        return TypeAll(loader.construct_scalar(node))

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar("!type/all", data.scope_name)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!type/all", TypeAll.constructor, loader)
        yaml.add_representer(TypeAll, TypeAll.representer)


class Step(Expect):
    """Sets an expectation for stepping behaviour, with the expected value being a list of integer lines:
    - !step exactly: while this !expect is active, we expect see exactly the expected lines in-order as many times as
      they appear in the expected lines list.
    - !step at_least: while this !expect is active, we expect to see each of the expected lines in-order at least as many
      times as they appear in the expected list, ignoring excess lines and lines not in the expected lines list.
    - !step never: while this !expect is active, we expect to not see any of the lines in the expected lines list.
    """

    def __init__(self, kind: str):
        self.kind = kind
        if kind not in ["exactly", "at_least", "never"]:
            raise DexterNodeError(self, f'invalid !step kind "{self.kind}"')

    def __repr__(self):
        return f"Step({self.kind})"

    @staticmethod
    def constructor(loader: yaml.Loader, node):
        return Step(loader.construct_scalar(node))

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar("!step", data.kind)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!step", Step.constructor, loader)
        yaml.add_representer(Step, Step.representer)


##############
## Execution Nodes: Can appear as leaf nodes directly under a state node to perform debugger actions when they become
## active, to advance the debugger state.


class Then:
    """Used to perform actions, such as finishing the test or continuing. Will trigger as soon as it becomes active, and
    intends to advance debugger state, so this must be used directly under a state node, e.g.:
    `!where {line: 4}: !then finish`
    """

    def __init__(self, command: str):
        self.command = command
        if not self.is_valid():
            raise DexterNodeError(self, f'Invalid !then command "{self.command}"')

    def is_valid(self) -> bool:
        valid_commands = ["finish", "step_out"]
        if self.command not in valid_commands:
            return False
        return True

    def __repr__(self):
        return f"Then({self.command})"

    @staticmethod
    def constructor(loader, node):
        return Then(loader.construct_scalar(node))

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar("!then", data.command)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!then", Then.constructor, loader)
        yaml.add_representer(Then, Then.representer)


##############
## Utility Nodes: Can be used anywhere in a script as a form of syntactic sugar.


class Address:
    """Named label for an address, which may resolve to different values with each test run, but will resolve
    consistently within a test run."""

    def __init__(self, name: str, offset: int):
        self.name = name
        self.offset = offset
        if not re.match(r"^([a-zA-Z_]\w*)$", name):
            raise DexterNodeError(self, f'Invalid !address identifier "{name}"')

    def __repr__(self):
        if not self.offset:
            offset_str = ""
        elif self.offset > 0:
            offset_str = f" + {self.offset}"
        else:
            offset_str = f" - {-self.offset}"
        return f"Address({self.name}{offset_str})"

    @staticmethod
    def constructor(loader, node):
        address_str = str(loader.construct_scalar(node)).strip()
        offset = 0
        if match := re.match(r"^([a-zA-Z_]\w*)\s*([+-])\s*(\d+)$", address_str):
            identifier, sign, number = match.groups()
            offset = int(number) if sign == "+" else -int(number)
            address_str = identifier
        return Address(address_str, offset)

    @staticmethod
    def representer(dumper, data: "Address"):
        if not data.offset:
            offset_str = ""
        elif data.offset > 0:
            offset_str = f"+{data.offset}"
        else:
            offset_str = f"-{-data.offset}"
        return dumper.represent_scalar("!address", data.name + offset_str)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!address", Address.constructor, loader)
        yaml.add_representer(Address, Address.representer)


@dataclass(frozen=True)
class Line:
    """Union class between an int or a Label, used to represent lines inside of Nodes."""

    value: Union[int, "Label"]

    def to_line(self, labels: FileLabels) -> int:
        if isinstance(self.value, int):
            return self.value
        return self.value.to_line(labels)

    def __repr__(self):
        return str(self.value)


@dataclass(frozen=True)
class DexRange:
    start: Line
    stop: Line

    def __repr__(self) -> str:
        return f"[{self.start} - {self.stop}]"

    # We use an inclusive range in Dexter scripts, while python ranges are exclusive.
    def to_range(self, labels: FileLabels) -> range:
        return range(self.start.to_line(labels), self.stop.to_line(labels) + 1)

    @staticmethod
    def constructor(loader: yaml.Loader, node):
        range_seq = loader.construct_sequence(node)
        if len(range_seq) != 2 or not all(
            isinstance(elt, (int, Label)) for elt in range_seq
        ):
            raise DexterNodeError(node, "range must have exactly 2 int elements")
        return DexRange(Line(range_seq[0]), Line(range_seq[1]))

    @staticmethod
    def representer(dumper, data: "DexRange"):
        return dumper.represent_sequence("!range", [data.start.value, data.stop.value])

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!range", DexRange.constructor, loader)
        yaml.add_representer(DexRange, DexRange.representer)


@dataclass(frozen=True)
class Label:
    name: str

    def to_line(self, file_labels: FileLabels) -> int:
        # Labels may contain offsets, which is accounted for here.
        raw_label = self.name.strip()
        label_str = raw_label
        offset = 0
        if match := re.match(r"^([a-zA-Z_]\w*)\s*([+-])\s*(\d+)$", raw_label):
            identifier, sign, number = match.groups()
            offset = int(number) if sign == "+" else -int(number)
            label_str = identifier
        if label_str not in file_labels.labels:
            raise DexterNodeError(
                self, f'Label "{label_str}" not found in file "{file_labels.file}"'
            )
        return file_labels.labels[label_str] + offset

    def __repr__(self):
        return f"Label({self.name})"

    @staticmethod
    def constructor(loader: yaml.Loader, node):
        return Label(loader.construct_scalar(node))

    @staticmethod
    def representer(dumper, data: "Label"):
        return dumper.represent_scalar("!label", data.name)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!label", Label.constructor, loader)
        yaml.add_representer(Label, Label.representer)


class Float:
    """Used to match against float values that may have an approximate range.
    There are four possible representations for a !float node, with/without a list of values and with/without a range:
    - `!float <value>` - checks for an exact match using floating point equality, e.g. !float 10 will match 10.0.
    - `!float <value>+-<range>` - checks for a match within the given range, e.g. !float 10 +- 0.2 will match any
                                  value in the range [9.8, 10.2]; range must also be a valid float value (and cannot be
                                  omitted if "+-" is passed).
    - `!float [<value>...] - checks for exact matches against any of the given values, e.g. !float [10, 11, 12] will
                             match 10.0, 11.0, or 12.0. This is effectively a shorthand for using a list of single float
                             values, e.g. `[!float 10, !float 11, !float 12]`.
    - `!float {values: [<value>...], range: <range>} - checks for matches against any of the given values, each of which
                                                       will match a range of +-<range>. As with normal lists, this is a
                                                       shorthand, e.g. !float{values: [1, 2], range: 0.1} is equivalent
                                                       to [!float 1 +- 0.1, !float 2 +- 0.1].
    """

    def __init__(self, values, range):
        try:
            if isinstance(values, list):
                values = [float(v) for v in values]
                range = float(range) if range is not None else None
            elif isinstance(values, str) and "+-" in values:
                assert (
                    range is None
                ), "Float has both an explicit range and a string-embedded range?"
                values, range = (float(n) for n in values.split("+-", maxsplit=1))
            else:
                assert range is None, "Explicit range passed with single float value?"
                values = float(values)
        except ValueError as err:
            raise DexterNodeError(self, f"!float received non-float value: {err}")
        self.values: Union[float, List[float]] = values
        self.range = range

    def __repr__(self):
        if self.range:
            return f"Float(values={self.values}, range={self.range})"
        return f"Float(values={self.values})"

    def _format_result(self, expected: float) -> str:
        """Formats an individual expected value for the purpose of comparing unique expected/seen values."""
        return (
            f"Float({expected}+-{self.range})"
            if self.range is not None
            else f"Float({expected})"
        )

    def get_expected_values(self) -> List[str]:
        """Returns a list of expected values in the form 'Float(<value>[+-<range>])'."""
        value_list = self.values if isinstance(self.values, list) else [self.values]
        return [self._format_result(v) for v in value_list]

    def matches(self, actual) -> Optional[str]:
        """If 'actual' matches this node, return a string representation of the matching value (in the same format as
        `get_expected_values` above), otherwise return None.
        """
        try:
            actual = float(actual)
        except ValueError:
            return None

        def float_match(expected: float) -> bool:
            if self.range is None:
                return expected == actual
            return abs(expected - actual) <= self.range

        if not isinstance(self.values, list):
            return (
                self._format_result(self.values) if float_match(self.values) else None
            )
        for expected in self.values:
            if float_match(expected):
                return self._format_result(expected)
        return None

    @staticmethod
    def constructor(loader, node):
        if isinstance(node, yaml.ScalarNode):
            # `!float <value>` or `!float <value> +- <range>`
            return Float(loader.construct_scalar(node), None)
        if isinstance(node, yaml.SequenceNode):
            # `!float [<value>...]`
            return Float(loader.construct_sequence(node), None)
        if isinstance(node, yaml.MappingNode):
            # `!float {values: [<value>...], range: <range>}`
            return Float(**loader.construct_mapping(node, deep=True))
        raise Exception("Invalid args to !float")

    @staticmethod
    def representer(dumper: yaml.Dumper, data):
        if data.range is None:
            if isinstance(data.values, list):
                return dumper.represent_sequence("!float", data.values, flow_style=True)
            return dumper.represent_scalar("!float", data.values)
        if not isinstance(data.values, list):
            return dumper.represent_scalar("!float", f"{data.values} +- {data.range}")
        mapping = {
            "values": data.values,
            "range": data.range,
        }
        return dumper.represent_mapping("!float", mapping, flow_style=True)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!float", Float.constructor, loader)
        yaml.add_representer(Float, Float.representer)
