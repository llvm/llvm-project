import argparse
from typing import TypedDict, Union, Optional, TextIO, NotRequired
from dataclasses import dataclass
import json


class Property(TypedDict):
    name: str
    type: str
    default: NotRequired[str]
    description: NotRequired[str]


class PropertyGroup(TypedDict):
    path: str
    """The full path to this group separated by dots"""
    properties: list[Property]


@dataclass
class PropertyTree:
    items: dict[str, Union["PropertyTree", Property]]


def append_group(tree: PropertyTree, group: PropertyGroup):
    segments = group["path"].split(".") if group["path"] else []

    subtree = tree
    for segment in segments:
        if segment not in subtree.items:
            subtree.items[segment] = PropertyTree(items={})
        subtree = subtree.items[segment]
        assert isinstance(subtree, PropertyTree)

    for property in group["properties"]:
        subtree.items[property["name"]] = property


def print_property(f: TextIO, path: str, property: Property):
    f.write(f"```{{lldbsetting}} {path}\n")
    f.write(f":type: \"{property['type']}\"\n\n")
    f.write(property.get("description", "").strip())
    f.write("\n\n")
    if "default" in property and property["default"]:
        f.write(f":default: {property['default']}\n")
    # FIXME: add enumerations (":enum {name}: {description}")
    f.write("```\n")


def print_tree(f: TextIO, level: int, prefix: str, name: str, tree: PropertyTree):
    if level > 0:
        f.write(f"{'#' * (level + 2)} {name}\n\n")

    leafs = sorted(
        filter(lambda it: isinstance(it[1], dict), tree.items.items()),
        key=lambda it: it[0],
    )
    for key, prop in leafs:
        assert isinstance(prop, dict)  # only needed for typing
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

This page lists all available settings in LLDB.
Settings can be set using `settings set <name> <value>`.
Values can be added to arrays and dictionaries with `settings append -- <name> <value>`.

## Root
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
            groups: list[PropertyGroup] = json.load(f)
        for group in groups:
            append_group(root, group)

    with open(args.output, "w") as f:
        f.write(HEADER)
        print_tree(f, 0, "", "", root)


if __name__ == "__main__":
    main()
