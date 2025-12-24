#!/usr/bin/env python3

"""
 Add attributes hook in an HLFIR code to test fir.call ModRef effects
 with the test-fir-alias-analysis-modref pass.

 This will insert mod ref test hook:
   - to any fir.call to a function which name starts with "test_effect_"
   - to any hlfir.declare for variable which name starts with "test_var_"
   - to any fir.box_addr - they are assigned box_addr_# hooks
"""

import sys
import re

box_addr_counter = 0

for line in sys.stdin:
    line = re.sub(
        r"(fir.call @_\w*P)(test_effect_\w*)(\(.*) : ",
        r'\1\2\3 {test.ptr ="\2"} : ',
        line,
    )
    line = re.sub(
        r'(hlfir.declare .*uniq_name =.*E)(test_var_\w*)"',
        r'\1\2", test.ptr ="\2"',
        line,
    )
    line, count = re.subn(
        r"(fir.box_addr.*) :",
        rf'\1 {{test.ptr ="box_addr_{box_addr_counter}"}} :',
        line,
    )
    if count > 0:
        box_addr_counter += 1
    sys.stdout.write(line)
