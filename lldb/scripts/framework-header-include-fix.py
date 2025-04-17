#!/usr/bin/env python3
# Usage: framework-header-include-fix.py <path/to/input-header.h> <path/to/output-header.h>
# This script modifies all #include lines in all lldb-rpc headers
# from either filesystem or local includes to liblldbrpc includes.

import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    input_path = str(args.input)
    output_path = str(args.output)
    with open(input_path, "r+") as input_file:
        lines = input_file.readlines()

    with open(output_path, "w+") as output_file:
        for line in lines:
            # Replace includes from RPCCommon to liblldbrpc includes.
            # e.g. #include <lldb-rpc/common/RPCArgument.h> -> #include <LLDBRPC/RPCArgument.h>
            if re.match(r".+<lldb-rpc/common", line):
                output_file.write(re.sub(r"<lldb-rpc/common", r"<LLDBRPC", line))
                # Replace all local file includes to liblldbrpc includes.
                # e.g. #include "SBFoo.h" -> #include <LLDBRPC/SBFoo.h>
            elif re.match(r'#include "(.*)"', line):
                include_filename = re.search(r'#include "(.*)"', line).groups()[0]
                output_file.write(
                    re.sub(
                        r'#include "(.*)"',
                        r"#include <LLDBRPC/" + include_filename + ">",
                        line,
                    )
                )
            else:
                # Write any line that doesn't need to be converted
                output_file.write(line)


if __name__ == "__main__":
    main()
