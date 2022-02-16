# RUN: %{python} %s

# Verify that each run of consecutive #include directives
# in each libcxx/include/ header is maintained in alphabetical order.

import glob
import os
import re


def exclude_from_consideration(path):
    return (
        path.endswith('.txt') or
        path.endswith('.modulemap') or
        os.path.basename(path) == '__config' or
        os.path.basename(path) == '__config_site.in' or
        os.path.basename(path) == '__libcpp_version' or
        os.path.basename(path) == '__locale' or
        not os.path.isfile(path)
    )


if __name__ == '__main__':
    libcxx_test_libcxx_lint = os.path.dirname(os.path.abspath(__file__))
    libcxx_include = os.path.abspath(os.path.join(libcxx_test_libcxx_lint, '../../../include'))
    assert os.path.isdir(libcxx_include)

    def pretty(path):
        return path[len(libcxx_include) + 1:]

    all_headers = [
        p for p in (
            glob.glob(os.path.join(libcxx_include, '*')) +
            glob.glob(os.path.join(libcxx_include, '__*/*.h'))
        ) if not exclude_from_consideration(p)
    ]

    okay = True
    for fname in all_headers:
        with open(fname, 'r') as f:
            lines = f.readlines()

        if '#  pragma GCC system_header\n' not in lines:
            if pretty(fname) not in ['__undef_macros']:
                okay = False
                print('FAILED TO FIND #  pragma GCC system_header in libcxx/include/%s!' % pretty(fname))
    assert okay
