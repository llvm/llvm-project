#!/usr/bin/env python
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
from header import HeaderFile
from class_implementation.classes.macro import Macro
from class_implementation.classes.type import Type
from class_implementation.classes.function import Function
from class_implementation.classes.include import Include
from class_implementation.classes.enumeration import Enumeration
from class_implementation.classes.object import Object


def yaml_to_classes(yaml_data):
    """
    Convert YAML data to header classes.

    Args:
        yaml_data: The YAML data containing header specifications.

    Returns:
        HeaderFile: An instance of HeaderFile populated with the data.
    """
    header_name = yaml_data.get("header")
    header = HeaderFile(header_name)

    for macro_data in yaml_data.get("macros", []):
        header.add_macro(Macro(macro_data["macro_name"], macro_data["macro_value"]))

    for type_data in yaml_data.get("types", []):
        header.add_type(Type(type_data["type_name"]))

    for enum_data in yaml_data.get("enums", []):
        header.add_enumeration(
            Enumeration(enum_data["name"], enum_data.get("value", None))
        )

    for function_data in yaml_data.get("functions", []):
        arguments = [arg["type"] for arg in function_data["arguments"]]
        guard = function_data.get("guard", None)
        attributes = function_data.get("attributes", None)
        standards = (function_data.get("standards", None),)
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

    for object_data in yaml_data.get("objects", []):
        header.add_object(
            Object(object_data["object_name"], object_data["object_type"])
        )

    for include_data in yaml_data.get("includes", []):
        header.add_include(Include(include_data))

    return header


def load_yaml_file(yaml_file):
    """
    Load YAML file and convert it to header classes.

    Args:
        yaml_file: The path to the YAML file.

    Returns:
        HeaderFile: An instance of HeaderFile populated with the data from the YAML file.
    """
    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_to_classes(yaml_data)


def fill_public_api(header_str, h_def_content):
    """
    Replace the %%public_api() placeholder in the .h.def content with the generated header content.

    Args:
        header_str: The generated header string.
        h_def_content: The content of the .h.def file.

    Returns:
        The final header content with the public API filled in.
    """
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

    yaml_data["functions"].append(function_dict)

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


def main(yaml_file, h_def_file, output_dir, add_function=None):
    """
    Main function to generate header files from YAML and .h.def templates.

    Args:
        yaml_file: Path to the YAML file containing header specification.
        h_def_file: Path to the .h.def template file.
        output_dir: Directory to output the generated header file.
        add_function: Details of the function to be added to the YAML file (if any).
    """

    if add_function:
        add_function_to_yaml(yaml_file, add_function)

    header = load_yaml_file(yaml_file)

    with open(h_def_file, "r") as f:
        h_def_content = f.read()

    header_str = str(header)
    final_header_content = fill_public_api(header_str, h_def_content)

    output_file_name = Path(h_def_file).stem
    output_file_path = Path(output_dir) / output_file_name

    with open(output_file_path, "w") as f:
        f.write(final_header_content)

    print(f"Generated header file: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate header files from YAML and .h.def templates"
    )
    parser.add_argument(
        "yaml_file", help="Path to the YAML file containing header specification"
    )
    parser.add_argument("h_def_file", help="Path to the .h.def template file")
    parser.add_argument(
        "--output_dir",
        default=".",
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
    args = parser.parse_args()

    main(args.yaml_file, args.h_def_file, args.output_dir, args.add_function)
