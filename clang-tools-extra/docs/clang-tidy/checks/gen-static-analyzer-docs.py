"""
Generates documentation based off the available static analyzers checks
References Checkers.td to determine what checks exist
"""

import subprocess
import json
import os
import re

"""Get path of script so files are always in correct directory"""
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

default_checkers_td_location = "../../../../clang/include/clang/StaticAnalyzer/Checkers/Checkers.td"
default_checkers_rst_location = "../../../../clang/docs/analyzer/checkers.rst"

"""Get dict of checker related info and parse for full check names

Returns:
  checkers: dict of checker info
"""
def get_checkers(checkers_td, checkers_rst):
    p = subprocess.Popen(
        [
            "llvm-tblgen",
            "--dump-json",
            "-I",
            os.path.dirname(checkers_td),
            checkers_td,
        ],
        stdout=subprocess.PIPE,
    )
    table_entries = json.loads(p.communicate()[0])
    documentable_checkers = []
    checkers = table_entries["!instanceof"]["Checker"]

    with open(checkers_rst, "r") as f:
        checker_rst_text = f.read()

    for checker_ in checkers:
        checker = table_entries[checker_]
        checker_name = checker["CheckerName"]
        package_ = checker["ParentPackage"]["def"]
        package = table_entries[package_]
        package_name = package["PackageName"]
        checker_package_prefix = package_name
        parent_package_ = package["ParentPackage"]
        hidden = (checker["Hidden"] != 0) or (package["Hidden"] != 0)

        while parent_package_ is not None:
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
            r"\.", "-", checker_package_prefix + "." + checker_name
        ).lower()

        if not hidden and "alpha" not in full_package_name.lower():
            checker["FullPackageName"] = full_package_name
            checker["ShortName"] = checker_package_prefix + "." + checker_name
            checker["AnchorUrl"] = anchor_url
            checker["Documentation"] = ".. _%s:" % (checker["ShortName"].replace(".","-")) in checker_rst_text
            documentable_checkers.append(checker)

    documentable_checkers.sort(key=lambda x: x["FullPackageName"])
    return documentable_checkers


"""Generate documentation for checker

Args:
  checker: Checker for which to generate documentation.
  has_documentation: Specify that there is other documentation to link to.
"""
def generate_documentation(checker, has_documentation):

    with open(
        os.path.join(__location__, "clang-analyzer", checker["ShortName"] + ".rst"), "w"
    ) as f:
        f.write(".. title:: clang-tidy - %s\n" % checker["FullPackageName"])
        if has_documentation:
            f.write(".. meta::\n")
            f.write(
                "   :http-equiv=refresh: 5;URL=https://clang.llvm.org/docs/analyzer/checkers.html#%s\n"
                % checker["AnchorUrl"]
            )
        f.write("\n")
        f.write("%s\n" % checker["FullPackageName"])
        f.write("=" * len(checker["FullPackageName"]) + "\n")
        help_text = checker["HelpText"].strip()
        if not help_text.endswith("."):
            help_text += "."
        characters = 80
        for word in help_text.split(" "):
            if characters+len(word)+1 > 80:
                characters = len(word)
                f.write("\n")
                f.write(word)
            else:
                f.write(" ")
                f.write(word)
                characters += len(word) + 1
        f.write("\n\n")
        if has_documentation:
            f.write(
                "The `%s` check is an alias, please see\n" % checker["FullPackageName"]
            )
            f.write(
                "`Clang Static Analyzer Available Checkers\n<https://clang.llvm.org/docs/analyzer/checkers.html#%s>`_\n"
                % checker["AnchorUrl"]
            )
            f.write("for more information.\n")
        else:
            f.write("The %s check is an alias of\nClang Static Analyzer %s.\n" % (checker["FullPackageName"], checker["ShortName"]));
        f.close()


"""Update list.rst to include the new checks

Args:
  checkers: dict acquired from get_checkers()
"""
def update_documentation_list(checkers):
    with open(os.path.join(__location__, "list.rst"), "r+") as f:
        f_text = f.read()
        check_text = f_text.split(':header: "Name", "Redirect", "Offers fixes"\n')[1]
        checks = [x for x in check_text.split("\n") if ":header:" not in x and x]
        old_check_text = "\n".join(checks)
        checks = [x for x in checks if "clang-analyzer-" not in x]
        for checker in checkers:
            if checker["Documentation"]:
                checks.append("   :doc:`%s <clang-analyzer/%s>`, `Clang Static Analyzer %s <https://clang.llvm.org/docs/analyzer/checkers.html#%s>`_," % (checker["FullPackageName"],
                                                        checker["ShortName"],  checker["ShortName"], checker["AnchorUrl"]))
            else:
                checks.append("   :doc:`%s <clang-analyzer/%s>`, Clang Static Analyzer %s," % (checker["FullPackageName"], checker["ShortName"],  checker["ShortName"]))

        checks.sort()

        # Overwrite file with new data
        f.seek(0)
        f_text = f_text.replace(old_check_text, "\n".join(checks))
        f.write(f_text)
        f.close()


def main():
    CheckersPath = os.path.join(__location__, default_checkers_td_location)
    if not os.path.exists(CheckersPath):
        print("Could not find Checkers.td under %s." % (os.path.abspath(CheckersPath)))
        exit(1)

    CheckersDoc = os.path.join(__location__, default_checkers_rst_location)
    if not os.path.exists(CheckersDoc):
        print("Could not find checkers.rst under %s." % (os.path.abspath(CheckersDoc)))
        exit(1)

    checkers = get_checkers(CheckersPath, CheckersDoc)
    for checker in checkers:
        generate_documentation(checker, checker["Documentation"])
        print("Generated documentation for: %s" % (checker["FullPackageName"]))
    update_documentation_list(checkers)


if __name__ == "__main__":
    main()
