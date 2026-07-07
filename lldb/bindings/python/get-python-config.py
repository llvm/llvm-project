#!/usr/bin/env python3

import os
import sys
import argparse
import sysconfig


def relpath_nodots(path, base):
    rel = os.path.normpath(os.path.relpath(path, base))
    assert not os.path.isabs(rel)
    parts = rel.split(os.path.sep)
    if parts and parts[0] == "..":
        raise ValueError(f"{path} is not under {base}")
    return rel


def main():
    parser = argparse.ArgumentParser(description="extract cmake variables from python")
    parser.add_argument("variable_name")
    parser.add_argument(
        "--stable-abi", action="store_true", help="Target the Stable C ABI"
    )
    args = parser.parse_args()
    if args.variable_name == "LLDB_PYTHON_RELATIVE_PATH":
        # LLDB_PYTHON_RELATIVE_PATH is the relative path from lldb's prefix
        # to where lldb's python libraries will be installed.
        #
        # This will always be lib/site-packages (or lib\site-packages).
        if os.name == "posix":
            print("lib/site-packages")
        elif os.name == "nt":
            print("Lib\\site-packages")
        else:
            raise
    elif args.variable_name == "LLDB_PYTHON_EXE_RELATIVE_PATH":
        tried = list()
        exe = sys.executable
        prefix = os.path.realpath(sys.prefix)
        while True:
            try:
                print(relpath_nodots(exe, prefix))
                break
            except ValueError:
                tried.append(exe)
                # Retry if the executable is symlinked or similar.
                # This is roughly equal to os.path.islink, except it also works for junctions on Windows.
                if os.path.realpath(exe) != exe:
                    exe = os.path.realpath(exe)
                    continue
                else:
                    print(
                        "Could not find a relative path to sys.executable under sys.prefix",
                        file=sys.stderr,
                    )
                    for e in tried:
                        print("tried:", e, file=sys.stderr)
                    print("realpath(sys.prefix):", prefix, file=sys.stderr)
                    print("sys.prefix:", sys.prefix, file=sys.stderr)
                    sys.exit(1)
    elif args.variable_name == "LLDB_PYTHON_EXT_SUFFIX":
        if args.stable_abi:
            shlib_suffix = sysconfig.get_config_var("SHLIB_SUFFIX")
            if shlib_suffix:
                print(".abi3%s" % shlib_suffix)
            else:
                assert os.name == "nt"
                if sysconfig.get_config_var("EXT_SUFFIX").startswith("_d"):
                    print("_d.pyd")
                else:
                    print(".pyd")
        else:
            print(sysconfig.get_config_var("EXT_SUFFIX"))
    else:
        parser.error(f"unknown variable {args.variable_name}")


if __name__ == "__main__":
    main()
