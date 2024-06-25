import yaml
import os
import re

from header import HeaderFile
from class_implementation.classes.macro import Macro
from class_implementation.classes.type import Type
from class_implementation.classes.function import Function
from class_implementation.classes.include import Include
from class_implementation.classes.enums import Enumeration
from class_implementation.classes.object import Object


def yaml_to_classes(yaml_data):
    header_name = yaml_data.get("header", "unknown.h")
    # standard = yaml_data.get('standard', None)
    header = HeaderFile(header_name)

    for macro_data in yaml_data.get("macros", []):
        header.add_macro(Macro(macro_data["macro_name"], macro_data["macro_value"]))

    for type_data in yaml_data.get("types", []):
        header.add_type(Type(type_data["type_name"]))

    for enum_data in yaml_data.get("enums", []):
        header.add_enumeration(
            Enumeration(enum_data["name"], enum_data.get("value", None))
        )

    for object_data in yaml_data.get("objects", []):
        header.add_object(
            Object(object_data["object_name"], object_data["object_type"])
        )

    for function_data in yaml_data.get("functions", []):
        arguments = [arg["type"] for arg in function_data["arguments"]]
        header.add_function(
            Function(
                function_data["return_type"],
                function_data["name"],
                arguments,
                function_data.get("guard"),
                function_data.get("attributes", []),
            )
        )

    for include_data in yaml_data.get("includes", []):
        header.add_include(Include(include_data))

    return header


def load_yaml_file(yaml_file):
    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_to_classes(yaml_data)


# will be used for specific functions a user wants to generate headers for
"""
def filter_functions(header, function_names):
    filtered_functions = []
    function_name_set = set(function_names)
    for func in header.functions:
        if func.name in function_name_set:
            filtered_functions.append(func)
    return filtered_functions
"""


def fill_public_api(header_str, h_def_content):
    # using regular expression to identify the public_api string
    return re.sub(r"%%public_api\(\)", header_str, h_def_content)


def main(yaml_file, h_def_file, output_dir):
    header = load_yaml_file(yaml_file)

    with open(h_def_file, "r") as f:
        h_def_content = f.read()

    header_str = str(header)
    final_header_content = fill_public_api(header_str, h_def_content)

    output_file_name = os.path.basename(h_def_file).replace(".def", "")
    output_file_path = os.path.join(output_dir, output_file_name)

    with open(output_file_path, "w") as f:
        f.write(final_header_content)

    print(f"Generated header file: {output_file_path}")


if __name__ == "__main__":
    import argparse

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
    args = parser.parse_args()

    main(args.yaml_file, args.h_def_file, args.output_dir)

# Example Command Line Arg: python3 yaml_to_classes.py yaml/stdc_stdbit.yaml h_def/stdbit.h.def --output_dir output
