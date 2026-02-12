import argparse
from typing import TypedDict, Union, Optional, TextIO, NotRequired
from dataclasses import dataclass
import json


PropertyDef = TypedDict(
    "PropertyDef",
    {
        "!superclasses": list[str],
        "Name": str,
        "Path": str,
        "Type": str,
        "Description": NotRequired[str],
        "HasDefaultUnsignedValue": NotRequired[int],
        "HasDefaultBooleanValue": NotRequired[int],
        "DefaultUnsignedValue": NotRequired[int],
        "HasDefaultStringValue": NotRequired[int],
        "DefaultStringValue": NotRequired[str],
        "HasDefaultEnumValue": NotRequired[int],
        "DefaultEnumValue": NotRequired[str],
        "EnumValues": NotRequired[str],
    },
)


class Property:
    name: str
    path: str
    type: str
    description: str
    default: Optional[str]

    def __init__(self, definition: PropertyDef):
        self.name = definition["Name"]
        self.path = definition["Path"]
        self.type = definition["Type"]
        self.description = definition.get("Description", "").strip()
        self.default = None

        has_default_unsigned = definition.get("HasDefaultUnsignedValue")
        has_default_bool = definition.get("HasDefaultBooleanValue")
        has_default_str = definition.get("HasDefaultStringValue")
        if has_default_bool == 1:
            assert has_default_unsigned
            self.default = (
                "true" if definition.get("DefaultUnsignedValue", 0) != 0 else "false"
            )
        elif has_default_unsigned:
            self.default = str(definition.get("DefaultUnsignedValue", 0))
        elif has_default_str:
            self.default = definition.get("DefaultStringValue")


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


def print_property(f: TextIO, path: str, property: Property):
    # Invoke lldbsetting directive. See MyST reference:
    # https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html
    f.write(f"```{{lldbsetting}} {path}\n")
    f.write(f':type: "{property.type}"\n\n')
    f.write(property.description)
    f.write("\n\n")
    if property.default:
        f.write(f":default: {property.default}\n")
    # FIXME: add enumerations (":enum {name}: {description}")
    f.write("```\n")


def print_tree(f: TextIO, level: int, prefix: str, name: str, tree: PropertyTree):
    if level > 0:
        f.write(f"{'#' * (level + 2)} {name}\n\n")

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
        prefix = f"{name}.{key}" if name else key
        print_tree(f, level + 1, prefix, key, subtree)


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

    root = PropertyTree(items={})
    for input in args.inputs:
        with open(input) as f:
            properties: dict[str, PropertyDef] = json.load(f)
        for key, prop in properties.items():
            if key.startswith("!"):
                continue  # tablegen metadata
            if "Property" not in prop["!superclasses"]:
                continue  # not a property
            append_property(root, Property(prop))

    with open(args.output, "w") as f:
        f.write(HEADER)
        print_tree(f, 0, "", "", root)


if __name__ == "__main__":
    main()
