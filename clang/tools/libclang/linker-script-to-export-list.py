import re
import os
import sys

input_file = open(sys.argv[1])
with open(sys.argv[2], "w") as output_file:
    for line in input_file:
        m = re.search('clang_[^;]+', line)
        if m:
            output_file.write(m.group(0) + '\n')
