#!/usr/bin/env python3
#
# ===- Generate headers for libc functions  -------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

import yaml
import argparse
from pathlib import Path

from enumeration import Enumeration
from function import Function
from gpu_headers import GpuHeaderFile as GpuHeader
from header import HeaderFile
from macro import Macro
from object import Object
from type import Type


def yaml_to_classes(yaml_data, header_class, entry_points=None):
    """
    Convert YAML data to header classes.

    Args:
        yaml_data: The YAML data containing header specifications.
        header_class: The class to use for creating the header.
        entry_points: A list of specific function names to include in the header.

    Returns:
        HeaderFile: An instance of HeaderFile populated with the data.
    """
    header_name = yaml_data.get("header")
    header = header_class(header_name)
    header.template_file = yaml_data.get("header_template")

    for macro_data in yaml_data.get("macros", []):
        header.add_macro(Macro(macro_data["macro_name"], macro_data["macro_value"]))

    types = yaml_data.get("types", [])
    sorted_types = sorted(types, key=lambda x: x["type_name"])
    for type_data in sorted_types:
        header.add_type(Type(type_data["type_name"]))

    for enum_data in yaml_data.get("enums", []):
        header.add_enumeration(
            Enumeration(enum_data["name"], enum_data.get("value", None))
        )

    functions = yaml_data.get("functions", [])
    if entry_points:
        entry_points_set = set(entry_points)
        functions = [f for f in functions if f["name"] in entry_points_set]
    sorted_functions = sorted(functions, key=lambda x: x["name"])
    guards = []
    guarded_function_dict = {}
    for function_data in sorted_functions:
        guard = function_data.get("guard", None)
        if guard is None:
            arguments = [arg["type"] for arg in function_data["arguments"]]
            attributes = function_data.get("attributes", None)
            standards = function_data.get("standards", None)
            header.add_function(
                Function(
                    function_data["return_type"],
                    function_data["name"],
                    arguments,
                    standards,
                    guard,
                    attributes,
                )
            )
        else:
            if guard not in guards:
                guards.append(guard)
                guarded_function_dict[guard] = []
                guarded_function_dict[guard].append(function_data)
            else:
                guarded_function_dict[guard].append(function_data)
    sorted_guards = sorted(guards)
    for guard in sorted_guards:
        for function_data in guarded_function_dict[guard]:
            arguments = [arg["type"] for arg in function_data["arguments"]]
            attributes = function_data.get("attributes", None)
            standards = function_data.get("standards", None)
            header.add_function(
                Function(
                    function_data["return_type"],
                    function_data["name"],
                    arguments,
                    standards,
                    guard,
                    attributes,
                )
            )

    objects = yaml_data.get("objects", [])
    sorted_objects = sorted(objects, key=lambda x: x["object_name"])
    for object_data in sorted_objects:
        header.add_object(
            Object(object_data["object_name"], object_data["object_type"])
        )

    return header


def load_yaml_file(yaml_file, header_class, entry_points):
    """
    Load YAML file and convert it to header classes.

    Args:
        yaml_file: Path to the YAML file.
        header_class: The class to use for creating the header (HeaderFile or GpuHeader).
        entry_points: A list of specific function names to include in the header.

    Returns:
        HeaderFile: An instance of HeaderFile populated with the data.
    """
    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_to_classes(yaml_data, header_class, entry_points)


def fill_public_api(header_str, h_def_content):
    """
    Replace the %%public_api() placeholder in the .h.def content with the generated header content.

    Args:
        header_str: The generated header string.
        h_def_content: The content of the .h.def file.

    Returns:
        The final header content with the public API filled in.
    """
    header_str = header_str.strip()
    return h_def_content.replace("%%public_api()", header_str, 1)


def parse_function_details(details):
    """
    Parse function details from a list of strings and return a Function object.

    Args:
        details: A list containing function details

    Returns:
        Function: An instance of Function initialized with the details.
    """
    return_type, name, arguments, standards, guard, attributes = details
    standards = standards.split(",") if standards != "null" else []
    arguments = [arg.strip() for arg in arguments.split(",")]
    attributes = attributes.split(",") if attributes != "null" else []

    return Function(
        return_type=return_type,
        name=name,
        arguments=arguments,
        standards=standards,
        guard=guard if guard != "null" else None,
        attributes=attributes if attributes else [],
    )


def add_function_to_yaml(yaml_file, function_details):
    """
    Add a function to the YAML file.

    Args:
        yaml_file: The path to the YAML file.
        function_details: A list containing function details (return_type, name, arguments, standards, guard, attributes).
    """
    new_function = parse_function_details(function_details)

    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)
    if "functions" not in yaml_data:
        yaml_data["functions"] = []

    function_dict = {
        "name": new_function.name,
        "standards": new_function.standards,
        "return_type": new_function.return_type,
        "arguments": [{"type": arg} for arg in new_function.arguments],
    }

    if new_function.guard:
        function_dict["guard"] = new_function.guard

    if new_function.attributes:
        function_dict["attributes"] = new_function.attributes

    insert_index = 0
    for i, func in enumerate(yaml_data["functions"]):
        if func["name"] > new_function.name:
            insert_index = i
            break
    else:
        insert_index = len(yaml_data["functions"])

    yaml_data["functions"].insert(insert_index, function_dict)

    class IndentYamlListDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(IndentYamlListDumper, self).increase_indent(flow, False)

    with open(yaml_file, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            Dumper=IndentYamlListDumper,
            default_flow_style=False,
            sort_keys=False,
        )

    print(f"Added function {new_function.name} to {yaml_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate header files from YAML")
    parser.add_argument(
        "yaml_file", help="Path to the YAML file containing header specification"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to output the generated header file",
    )
    parser.add_argument(
        "--add_function",
        nargs=6,
        metavar=(
            "RETURN_TYPE",
            "NAME",
            "ARGUMENTS",
            "STANDARDS",
            "GUARD",
            "ATTRIBUTES",
        ),
        help="Add a function to the YAML file",
    )
    parser.add_argument(
        "--entry-point",
        action="append",
        help="Entry point to include",
        dest="entry_points",
    )
    parser.add_argument(
        "--export-decls",
        action="store_true",
        help="Flag to use GpuHeader for exporting declarations",
    )
    args = parser.parse_args()

    if args.add_function:
        add_function_to_yaml(args.yaml_file, args.add_function)

    header_class = GpuHeader if args.export_decls else HeaderFile
    header = load_yaml_file(args.yaml_file, header_class, args.entry_points)

    header_str = str(header)

    if args.output_dir:
        output_file_path = Path(args.output_dir)
        if output_file_path.is_dir():
            output_file_path /= f"{Path(args.yaml_file).stem}.h"
    else:
        output_file_path = Path(f"{Path(args.yaml_file).stem}.h")

    if args.export_decls:
        with open(output_file_path, "w") as f:
            f.write(header_str)


if __name__ == "__main__":
    main()
