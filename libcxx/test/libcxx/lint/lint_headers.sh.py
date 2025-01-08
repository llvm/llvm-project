# RUN: %{python} %s

# Verify that each run of consecutive #include directives
# in each libcxx/include/ header is maintained in alphabetical order.

import glob
import os
import re


def exclude_from_consideration(path):
    return (
        path.endswith(".txt")
        or path.endswith(".modulemap")
        or os.path.basename(path) == "__config"
        or os.path.basename(path) == "__config_site.in"
        or os.path.basename(path) == "libcxx.imp"
        or os.path.basename(path).startswith("__pstl")
        or not os.path.isfile(path)  # TODO: Remove once PSTL integration is finished
    )


def check_for_pragma_GCC_system_header(pretty_fname, lines):
    if pretty_fname not in ["__undef_macros"]:
        for line in lines:
            if re.match("# *pragma GCC system_header\n", line):
                return True
        print(
            "FAILED TO FIND #  pragma GCC system_header in libcxx/include/%s"
            % pretty_fname
        )
        return False
    return True


if __name__ == "__main__":
    libcxx_test_libcxx_lint = os.path.dirname(os.path.abspath(__file__))
    libcxx_include = os.path.abspath(
        os.path.join(libcxx_test_libcxx_lint, "../../../include")
    )
    assert os.path.isdir(libcxx_include)

    def pretty(path):
        return path[len(libcxx_include) + 1 :]

    all_headers = [
        p
        for p in (
            glob.glob(os.path.join(libcxx_include, "*"))
            + glob.glob(os.path.join(libcxx_include, "__*/*.h"))
        )
        if not exclude_from_consideration(p)
    ]

    okay = True
    for fname in all_headers:
        pretty_fname = pretty(fname)
        with open(fname, "r") as f:
            lines = f.readlines()

        okay = check_for_pragma_GCC_system_header(pretty_fname, lines) and okay

    assert okay
