# RUN: ld.lld --help 2>&1 | FileCheck --strict-whitespace %s
# CHECK:  OPTIONS:
# CHECK:  --output=<value>        - Alias for -o
# CHECK:  --output <value>        - Alias for -o
# CHECK:  -o <path>               - Path to file to write output
