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
from typing import Any, Dict, Optional, Union
import yaml
from dex.dextIR.ValueIR import ValueIR
from dex.utils.Exceptions import Error


def setup_yaml_parser(loader):
    reg_classes = [
        Where,
        Value,
        DexRange,
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


###################
## Structural Nodes: These are used as keys in the Script, and collectively define Dexter's actions when running a test:
##                   how it steps and navigates through the debuggee program, and what information it collects from the
##                   debugger.


class Where:
    """One or more instances of this class define a range of steps in a debugging session. Any expects in the script
    within scope of a "Where" will only be evaluated for the steps where the Where applies.
    """

    def __init__(self, attributes: dict):
        self.file: Optional[str] = attributes.pop("file", None)
        self.function: Union[list[str], str, None] = attributes.pop("function", None)
        self.lines: Union[int, DexRange, None] = attributes.pop("lines", None)
        self.after_hit_count: Optional[int] = attributes.pop("after_hit_count", None)
        self.for_hit_count: Optional[int] = attributes.pop("for_hit_count", None)
        self.conditions: dict = attributes.pop("conditions", None)
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
        return f"Where(" + ", ".join(elts) + ")"

    def get_attrs(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "function": self.function,
            "lines": self.lines,
            "for_hit_count": self.for_hit_count,
            "after_hit_count": self.after_hit_count,
            "conditions": self.conditions,
        }

    @staticmethod
    def constructor(loader: yaml.Loader, node):
        return Where(loader.construct_mapping(node))

    @staticmethod
    def representer(dumper: yaml.Dumper, data: "Where"):
        mapping = {
            name: value for name, value in data.get_attrs().items() if value is not None
        }
        return dumper.represent_mapping("!where", mapping, flow_style=True)

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!where", Where.constructor, loader)
        yaml.add_representer(Where, Where.representer)

    def get_lines(self) -> range:
        """Returns the range of line numbers that this Where references, returning an empty range if this Where does not
        refer to any lines."""
        if not self.lines:
            return range(-1)
        if isinstance(self.lines, int):
            return range(self.lines, self.lines + 1)
        assert isinstance(
            self.lines, DexRange
        ), f"Invalid type for lines: {self.lines}: ({type(self.lines)})"
        return self.lines.to_range()


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
class DexRange:
    start: int
    stop: int

    def __repr__(self) -> str:
        return f"[{self.start} - {self.stop}]"

    # We use an inclusive range in Dexter scripts, while python ranges are exclusive.
    def to_range(self) -> range:
        return range(self.start, self.stop + 1)

    @staticmethod
    def constructor(loader: yaml.Loader, node):
        range_seq = loader.construct_sequence(node)
        if len(range_seq) != 2 or not all(isinstance(elt, int) for elt in range_seq):
            raise DexterNodeError(node, "range must have exactly 2 int elements")
        return DexRange(range_seq[0], range_seq[1])

    @staticmethod
    def representer(dumper, data: "DexRange"):
        return dumper.represent_sequence("!range", [data.start, data.stop])

    @staticmethod
    def register_yaml(loader):
        yaml.add_constructor("!range", DexRange.constructor, loader)
        yaml.add_representer(DexRange, DexRange.representer)
