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
from typing import Any, Dict, Optional, Union
import yaml
from dex.dextIR.ValueIR import ValueIR
from dex.utils.Exceptions import Error


def setup_yaml_parser(loader):
    reg_classes = [
        Where,
        Value,
        DexRange,
        Label,
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
    """

    def __init__(self, attributes: dict, is_and: bool):
        self.file: Optional[str] = attributes.pop("file", None)
        self.function: Union[list[str], str, None] = attributes.pop("function", None)
        lines = attributes.pop("lines", None)
        if isinstance(lines, (int, Label)):
            lines = Line(lines)
        self.lines: Union[Line, DexRange, None] = lines
        self.after_hit_count: Optional[int] = attributes.pop("after_hit_count", None)
        self.for_hit_count: Optional[int] = attributes.pop("for_hit_count", None)
        self.conditions: dict = attributes.pop("conditions", None)
        self.is_and = is_and
        if attributes:
            raise DexterNodeError(
                self, f"unexpected attributes {', '.join(attributes)}"
            )
        if (
            not self.function
            and not self.lines
            and (self.for_hit_count or self.after_hit_count)
        ):
            raise DexterNodeError(
                self, "can't check hit counts without an explicit lines or function arg"
            )

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

    @abc.abstractmethod
    def get_watched_expr(self) -> str:
        """Returns the list of expressions that this Expect wants to evaluate."""


class Value(Expect):
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


##############
## Utility Nodes: Can be used anywhere in a script as a form of syntactic sugar.


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
