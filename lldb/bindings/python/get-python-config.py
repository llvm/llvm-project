#!/usr/bin/env python3

import os
import sys
import argparse
import sysconfig
import distutils.sysconfig


def relpath_nodots(path, base):
    rel = os.path.normpath(os.path.relpath(path, base))
    assert not os.path.isabs(rel)
    parts = rel.split(os.path.sep)
    if parts and parts[0] == '..':
        raise ValueError(f"{path} is not under {base}")
    return rel

def main():
    parser = argparse.ArgumentParser(description="extract cmake variables from python")
    parser.add_argument("variable_name")
    args = parser.parse_args()
    if args.variable_name == "LLDB_PYTHON_RELATIVE_PATH":
        print(distutils.sysconfig.get_python_lib(True, False, ''))
    elif args.variable_name == "LLDB_PYTHON_EXE_RELATIVE_PATH":
        tried = list()
        exe = sys.executable
        while True:
            try:
                print(relpath_nodots(exe, sys.prefix))
                break
            except ValueError:
                tried.append(exe)
                if os.path.islink(exe):
                    exe = os.path.join(os.path.dirname(exe), os.readlink(exe))
                    continue
                else:
                    print("Could not find a relative path to sys.executable under sys.prefix", file=sys.stderr)
                    for e in tried:
                        print("tried:", e, file=sys.stderr)
                    print("sys.prefix:", sys.prefix, file=sys.stderr)
                    sys.exit(1)
    elif args.variable_name == "LLDB_PYTHON_EXT_SUFFIX":
        print(sysconfig.get_config_var('EXT_SUFFIX'))
    else:
        parser.error(f"unknown variable {args.variable_name}")

if __name__ == '__main__':
    main()