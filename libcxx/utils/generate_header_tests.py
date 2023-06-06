#!/usr/bin/env python

import contextlib
import glob
import io
import os
import pathlib
import re

import libcxx.test.header_information


def find_script(file):
    """Finds the script used to generate a file inside the file itself. The script is delimited by
    BEGIN-SCRIPT and END-SCRIPT markers.
    """
    with open(file, "r") as f:
        content = f.read()

    match = re.search(
        r"^BEGIN-SCRIPT$(.+)^END-SCRIPT$", content, flags=re.MULTILINE | re.DOTALL
    )
    if not match:
        raise RuntimeError(
            "Was unable to find a script delimited with BEGIN-SCRIPT/END-SCRIPT markers in {}".format(
                test_file
            )
        )
    return match.group(1)


def execute_script(script, variables):
    """Executes the provided Mako template with the given variables available during the
    evaluation of the script, and returns the result.
    """
    code = compile(script, "fake-filename", "exec")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(code, variables)
        output = output.getvalue()
    return output


def generate_new_file(file, new_content):
    """Generates the new content of the file by inserting the new content in-between
    two '// GENERATED-MARKER' markers located in the file.
    """
    with open(file, "r") as f:
        old_content = f.read()

    try:
        before, begin_marker, _, end_marker, after = re.split(
            r"(// GENERATED-MARKER\n)", old_content, flags=re.MULTILINE | re.DOTALL
        )
    except ValueError:
        raise RuntimeError(
            "Failed to split {} based on markers, please make sure the file has exactly two '// GENERATED-MARKER' occurrences".format(
                file
            )
        )

    return before + begin_marker + new_content + end_marker + after


def produce(test_file, variables):
    script = find_script(test_file)
    result = execute_script(script, variables)
    new_content = generate_new_file(test_file, result)
    with open(test_file, "w", newline="\n") as f:
        f.write(new_content)


def main():
    monorepo_root = pathlib.Path(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    test = pathlib.Path(os.path.join(monorepo_root, "libcxx", "test"))
    assert monorepo_root.exists()

    produce(test.joinpath("libcxx/clang_tidy.sh.cpp"), libcxx.test.header_information.variables)
    produce(test.joinpath("libcxx/double_include.sh.cpp"), libcxx.test.header_information.variables)
    produce(test.joinpath("libcxx/min_max_macros.compile.pass.cpp"), libcxx.test.header_information.variables)
    produce(test.joinpath("libcxx/modules_include.sh.cpp"), libcxx.test.header_information.variables)
    produce(test.joinpath("libcxx/nasty_macros.compile.pass.cpp"), libcxx.test.header_information.variables)
    produce(test.joinpath("libcxx/no_assert_include.compile.pass.cpp"), libcxx.test.header_information.variables)
    produce(test.joinpath("libcxx/private_headers.verify.cpp"), libcxx.test.header_information.variables)
    produce(test.joinpath("libcxx/transitive_includes.sh.cpp"), libcxx.test.header_information.variables)


if __name__ == "__main__":
    main()
