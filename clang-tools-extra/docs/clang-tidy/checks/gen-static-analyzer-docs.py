"""
Generates documentation based off the available static analyzers checks
References Checkers.td to determine what checks exist
"""

import argparse
import subprocess
import json
import os
import re

"""Get path of script so files are always in correct directory"""
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

"""Get dict of checker related info and parse for full check names

Returns:
  checkers: dict of checker info
"""


def get_checkers(checkers_td_directory):
    p = subprocess.Popen(
        [
            "llvm-tblgen",
            "--dump-json",
            "-I",
            checkers_td_directory,
            checkers_td_directory + "Checkers.td",
        ],
        stdout=subprocess.PIPE,
    )
    table_entries = json.loads(p.communicate()[0])
    documentable_checkers = []
    checkers = table_entries["!instanceof"]["Checker"]
    packages = table_entries["!instanceof"]["Package"]

    for checker_ in checkers:
        checker = table_entries[checker_]
        checker_name = checker["CheckerName"]
        package_ = checker["ParentPackage"]["def"]
        package = table_entries[package_]
        package_name = package["PackageName"]
        checker_package_prefix = package_name
        parent_package_ = package["ParentPackage"]
        hidden = (checker["Hidden"] != 0) or (package["Hidden"] != 0)

        while parent_package_ != None:
            parent_package = table_entries[parent_package_["def"]]
            checker_package_prefix = (
                parent_package["PackageName"] + "." + checker_package_prefix
            )
            hidden = hidden or parent_package["Hidden"] != 0
            parent_package_ = parent_package["ParentPackage"]

        full_package_name = (
            "clang-analyzer-" + checker_package_prefix + "." + checker_name
        )
        anchor_url = re.sub(
            "\.", "-", checker_package_prefix + "." + checker_name
        ).lower()

        if not hidden and "alpha" not in full_package_name.lower():
            checker["FullPackageName"] = full_package_name
            checker["AnchorUrl"] = anchor_url
            documentable_checkers.append(checker)

    documentable_checkers.sort(key=lambda x: x["FullPackageName"])
    return documentable_checkers


"""Generate documentation for checker

Args:
  checker: Checker for which to generate documentation.
  only_help_text: Generate documentation based off the checker description.
    Used when there is no other documentation to link to.
"""


def generate_documentation(checker, only_help_text=False):
    with open(
        os.path.join(__location__, checker["FullPackageName"] + ".rst"), "w"
    ) as f:
        f.write(".. title:: clang-tidy - %s\n" % checker["FullPackageName"])
        if not only_help_text:
            f.write(".. meta::\n")
            f.write(
                "   :http-equiv=refresh: 5;URL=https://clang.llvm.org/docs/analyzer/checkers.html#%s\n"
                % checker["AnchorUrl"]
            )
        f.write("\n")
        f.write("%s\n" % checker["FullPackageName"])
        f.write("=" * len(checker["FullPackageName"]) + "\n")
        f.write("\n")
        if only_help_text:
            f.write("%s\n" % checker["HelpText"])
        else:
            f.write(
                "The %s check is an alias, please see\n" % checker["FullPackageName"]
            )
            f.write(
                "`Clang Static Analyzer Available Checkers <https://clang.llvm.org/docs/analyzer/checkers.html#%s>`_\n"
                % checker["AnchorUrl"]
            )
            f.write("for more information.\n")
        f.close()


"""Update list.rst to include the new checks

Args:
  checkers: dict acquired from get_checkers()
"""


def update_documentation_list(checkers):
    with open(os.path.join(__location__, "list.rst"), "r+") as f:
        f_text = f.read()
        header, check_text = f_text.split(".. toctree::\n")
        checks = check_text.split("\n")
        for checker in checkers:
            if ("   %s" % checker["FullPackageName"]) not in checks:
                checks.append("   %s" % checker["FullPackageName"])
        checks.sort()

        # Overwrite file with new data
        f.seek(0)
        f.write(header)
        f.write(".. toctree::")
        for check in checks:
            f.write("%s\n" % check)
        f.close()


default_path_monorepo = "../../../../clang/include/clang/StaticAnalyzer/Checkers/"
default_path_in_tree = "../../../../../include/clang/StaticAnalyzer/Checkers/"


def parse_arguments():
    """Set up and parse command-line arguments
    Returns:
      file_path: Path to Checkers.td"""
    usage = """Parse Checkers.td to generate documentation for static analyzer checks"""
    parse = argparse.ArgumentParser(description=usage)

    file_path_help = """Path to Checkers directory
                    defaults to ../../../../clang/include/clang/StaticAnalyzer/Checkers/ if it exists
                    then to ../../../../../include/clang/StaticAnalyzer/Checkers/"""

    default_path = None
    if os.path.exists(default_path_monorepo):
        default_path = default_path_monorepo
    elif os.path.exists(default_path_in_tree):
        default_path = default_path_in_tree

    parse.add_argument(
        "file", type=str, help=file_path_help, nargs="?", default=default_path
    )
    args = parse.parse_args()

    if args.file is None:
        print("Could not find Checkers directory. Please see -h")
        exit(1)

    return args.file


def main():
    file_path = parse_arguments()
    checkers = get_checkers(file_path)
    for checker in checkers:
        # No documentation nor alpha documentation
        if checker["Documentation"][1] == 0 and checker["Documentation"][0] == 0:
            generate_documentation(checker, True)
        else:
            generate_documentation(checker)
        print("Generated documentation for: %s" % checker["FullPackageName"])
    update_documentation_list(checkers)


if __name__ == "__main__":
    main()
