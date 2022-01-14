# RUN: %{python} %s

# Verify that each list of private submodules in libcxx/include/module.modulemap
# is maintained in alphabetical order.

import os
import re


if __name__ == '__main__':
    libcxx_test_libcxx_lint = os.path.dirname(os.path.abspath(__file__))
    libcxx = os.path.abspath(os.path.join(libcxx_test_libcxx_lint, '../../..'))
    modulemap_name = os.path.join(libcxx, 'include/module.modulemap')
    assert os.path.isfile(modulemap_name)

    okay = True
    prevline = None
    with open(modulemap_name, 'r') as f:
        for line in f.readlines():
            if re.match(r'^\s*module.*[{]\s*private', line):
                if (prevline is not None) and (line < prevline):
                    okay = False
                    print('LINES OUT OF ORDER in libcxx/include/module.modulemap!')
                    print(prevline)
                    print(line)
                prevline = line
            else:
                prevline = None
    assert okay
