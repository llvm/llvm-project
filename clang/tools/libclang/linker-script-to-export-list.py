import re
import os
import sys

input_file = open(sys.argv[1])
output_file = open(sys.argv[2])

for line in input_file:
    m = re.search('clang_[^;]+', line)
    if m:
        output_file.write(m.group(0))
