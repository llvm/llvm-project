import argparse
from typing import Any, Dict, TypedDict, Union, Optional, TextIO
from dataclasses import dataclass
import json
import re

PropertyDef = Dict[str, Any]
DefTree = Dict[str, Any]
EnumValueDef = Dict[str, Any]
EnumDefDict = Dict[str, Any]


@dataclass
class EnumValue:
    value: str
    name: str
    description: str

    def __init__(self, definition: EnumValueDef, prefix: str):
        self.value = prefix + definition["Value"]
        self.name = definition["Name"]
        self.description = definition["Description"]

    def matches(self, value: str):
        return self.value.removeprefix("lldb::") == value.removeprefix("lldb::")


@dataclass
class EnumDef:
    items: list[EnumValue]

    def __init__(self, definition: EnumDefDict, tree: DefTree):
        prefix = definition.get("Prefix", "")
        self.items = [EnumValue(tree[d["def"]], prefix) for d in definition["Values"]]


class Property:
    name: str
    path: str
    type: str
    description: str
    default: Optional[str]
    enum_name: Optional[str]
    enum_values: Optional[EnumDef]

    def __init__(self, definition: PropertyDef):
        self.name = definition["Name"]
        self.path = definition["Path"]
        self.type = definition["Type"]
        self.description = definition.get("Description", "").strip()
        self.default = None
        self.enum_name = None
        self.enum_values = None

        has_default_unsigned = definition.get("HasDefaultUnsignedValue")
        has_default_bool = definition.get("HasDefaultBooleanValue")
        has_default_str = definition.get("HasDefaultStringValue")
        has_default_enum = definition.get("HasDefaultEnumValue")
        if has_default_bool == 1:
            assert has_default_unsigned
            self.default = (
                "true" if definition.get("DefaultUnsignedValue", 0) != 0 else "false"
            )
        elif has_default_unsigned:
            self.default = str(definition.get("DefaultUnsignedValue", 0))
        elif has_default_str:
            self.default = definition.get("DefaultStringValue")
        elif has_default_enum and self.type != "Language":
            self.default = definition.get("DefaultEnumValue")

        enum_values: Optional[str] = definition.get("EnumValues")
        if enum_values:
            self.enum_name = enum_values.removeprefix("OptionEnumValues(").removesuffix(
                ")"
            )

    def find_enum_values(self, enums: Dict[str, EnumDef]):
        if self.enum_name is None:
            return

        self.enum_values = enums[self.enum_name]
        if self.default is not None:
            default = next(
                (v.name for v in self.enum_values.items if v.matches(self.default)),
                None,
            )
            if not default:
                raise RuntimeError(
                    f"Failed to find enum value for {self.default} in {self.enum_values}"
                )
            self.default = default


class PropertyGroup(TypedDict):
    path: str
    """The full path to this group separated by dots (e.g. 'target.process')"""
    properties: list[Property]


@dataclass
class PropertyTree:
    items: dict[str, Union["PropertyTree", Property]]


def append_property(tree: PropertyTree, prop: Property):
    segments = prop.path.split(".") if prop.path else []

    subtree = tree
    for segment in segments:
        if segment not in subtree.items:
            subtree.items[segment] = PropertyTree(items={})
        subtree = subtree.items[segment]
        assert isinstance(subtree, PropertyTree)

    subtree.items[prop.name] = prop


def wrap_inline_code(text: str):
    n_backticks = max([len(s) for s in re.findall("`+", text)], default=0)
    fence = "`" * (n_backticks + 1)
    if text.startswith("`") or text.endswith("`"):
        text = f" {text} "
    return f"{fence}{text}{fence}"


def print_property(f: TextIO, path: str, property: Property):
    # Invoke lldbsetting directive (lldb/docs/_ext/lldb_setting.py)
    f.write(f"```{{lldbsetting}} {path}\n")
    f.write(f':type: "{property.type}"\n\n')
    f.write(property.description)
    f.write("\n\n")
    if property.enum_values:
        for enum in property.enum_values.items:
            f.write(f":enum {enum.name}: {enum.description}\n")
    if property.default:
        f.write(f":default: {wrap_inline_code(property.default)}\n")
    # FIXME: add enumerations (":enum {name}: {description}")
    f.write("```\n")


def print_tree(f: TextIO, level: int, prefix: str, name: str, tree: PropertyTree):
    if level > 0:
        f.write(f"{'#' * (level + 1)} {name}\n\n")

    leafs = sorted(
        filter(lambda it: isinstance(it[1], Property), tree.items.items()),
        key=lambda it: it[0],
    )
    for key, prop in leafs:
        assert isinstance(prop, Property)  # only needed for typing
        path = f"{prefix}.{key}" if prefix else key
        print_property(f, path, prop)

    groups = sorted(
        filter(lambda it: isinstance(it[1], PropertyTree), tree.items.items()),
        key=lambda it: it[0],
    )
    for key, subtree in groups:
        assert isinstance(subtree, PropertyTree)  # only needed for typing
        sub_prefix = f"{prefix}.{key}" if prefix else key
        print_tree(f, level + 1, sub_prefix, key, subtree)


HEADER = """
# Settings

This page lists all possible settings in LLDB.
Settings can be set using `settings set <name> <value>`.
Values can be added to arrays and dictionaries with `settings append -- <name> <value>`.

```{note}
Some settings only exist for particular LLDB build configurations and so will
not be present in all copies of LLDB.
```
"""


def main():
    parser = argparse.ArgumentParser(
        prog="gen-property-docs-from-json",
        description="Generate Markdown from multiple property docs",
    )
    parser.add_argument("-o", "--output", help="Path to output file")
    parser.add_argument("inputs", nargs="*")
    args = parser.parse_args()

    all_properties: list[Property] = []
    enums = dict[str, EnumDef]()

    for input in args.inputs:
        with open(input, encoding="utf-8") as f:
            properties: dict[str, PropertyDef] = json.load(f)
        for key, prop in properties.items():
            if key.startswith("!"):
                continue  # tablegen metadata
            superclasses = prop["!superclasses"]
            if "Property" in superclasses:
                all_properties.append(Property(prop))
            if "EnumDef" in superclasses:
                enums["g_" + key] = EnumDef(prop, properties)

    root = PropertyTree(items={})
    for prop in all_properties:
        prop.find_enum_values(enums)
        append_property(root, prop)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(HEADER)
        print_tree(f, 0, "", "", root)


if __name__ == "__main__":
    main()
