import argparse
import os
import subprocess
import sys

DEFAULT_CPU_COUNT = 4

def do_compile(args):
    ret = False

    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = DEFAULT_CPU_COUNT

    make_cmd = ["make", "-j", str(cpu_count), "sycl-toolchain"]
    print(make_cmd)

    subprocess.check_call(make_cmd, cwd=args.obj_dir)

    ret = True
    return ret

def main():
    parser = argparse.ArgumentParser(prog="compile.py",
                                     description="script to do compile",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--build-number", metavar="BUILD_NUM", help="build number")
    parser.add_argument("-b", "--branch", metavar="BRANCH", help="pull request branch")
    parser.add_argument("-d", "--base-branch", metavar="BASE_BRANCH", help="pull request base branch")
    parser.add_argument("-r", "--pr-number", metavar="PR_NUM", help="pull request number")
    parser.add_argument("-w", "--builder-dir", metavar="BUILDER_DIR",
                        help="builder directory, which is the directory contains source and build directories")
    parser.add_argument("-s", "--src-dir", metavar="SRC_DIR", help="source directory")
    parser.add_argument("-o", "--obj-dir", metavar="OBJ_DIR", required=True, help="build directory")

    args = parser.parse_args()

    print("args:{}".format(args))

    return do_compile(args)

if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)

