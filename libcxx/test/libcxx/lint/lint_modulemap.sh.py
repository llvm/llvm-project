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
                # Check that these lines are all of the expected format.
                # This incidentally checks for typos in the module name.
                if re.match(r'^\s*module (\w+)\s+[{] private header "\1(.h)?"\s+export [*] [}]', line):
                    # It's a top-level private header, such as <__bit_reference>.
                    pass
                elif re.match(r'^\s*module (\w+)\s+[{] private header "__\w+/\1[.]h" [}]', line):
                    # It's a private submodule, such as <__utility/swap.h>.
                    pass
                elif re.match(r'^\s*module (\w+)_fwd\s+[{] private header "__fwd/\1[.]h" [}]', line):
                    # It's a private submodule with forward declarations, such as <__fwd/span.h>.
                    pass
                else:
                    okay = False
                    print("LINE DOESN'T MATCH REGEX in libcxx/include/module.modulemap!")
                    print(line)
                # Check that these lines are alphabetized.
                if (prevline is not None) and (line < prevline):
                    okay = False
                    print('LINES OUT OF ORDER in libcxx/include/module.modulemap!')
                    print(prevline)
                    print(line)
                prevline = line
            else:
                prevline = None
    assert okay
