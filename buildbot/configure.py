import argparse
import os
import subprocess
import sys

def do_configure(args):
    ret = False

    llvm_dir = os.path.join(args.src_dir, "llvm")
    sycl_dir = os.path.join(args.src_dir, "sycl")
    spirv_dir = os.path.join(args.src_dir, "llvm-spirv")
    ocl_header_dir = os.path.join(args.obj_dir, "OpenCL-Headers")
    icd_loader_lib = os.path.join(args.obj_dir, "OpenCL-ICD-Loader", "build", "libOpenCL.so")

    cmake_cmd = ["cmake",
                 "-DCMAKE_BUILD_TYPE={}".format(args.build_type),
                 "-DLLVM_ENABLE_PROJECTS=clang",
                 "-DLLVM_EXTERNAL_SYCL_SOURCE_DIR={}".format(sycl_dir),
                 "-DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR={}".format(spirv_dir),
                 "-DLLVM_TOOL_SYCL_BUILD=ON",
                 "-DLLVM_TOOL_LLVM_SPIRV_BUILD=ON",
                 "-DOpenCL_INCLUDE_DIR={}".format(ocl_header_dir),
                 "-DOpenCL_LIBRARY={}".format(icd_loader_lib),
                 "-DLLVM_BUILD_TOOLS=OFF",
                 llvm_dir]

    print(cmake_cmd)

    try:
        subprocess.check_call(cmake_cmd, cwd=args.obj_dir)
    except subprocess.CalledProcessError:
        cmake_cache = os.path.join(args.obj_dir, "CMakeCache.txt")
        if os.path.isfile(cmake_cache):
            os.remove(cmake_cache)
        subprocess.check_call(cmake_cmd, cwd=args.obj_dir)

    ret = True
    return ret

def main():
    parser = argparse.ArgumentParser(prog="configure.py",
                                     description="script to do configure",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--build-number", metavar="BUILD_NUM", help="build number")
    parser.add_argument("-b", "--branch", metavar="BRANCH", help="pull request branch")
    parser.add_argument("-d", "--base-branch", metavar="BASE_BRANCH", help="pull request base branch")
    parser.add_argument("-r", "--pr-number", metavar="PR_NUM", help="pull request number")
    parser.add_argument("-w", "--builder-dir", metavar="BUILDER_DIR",
                        help="builder directory, which is the directory contains source and build directories")
    parser.add_argument("-s", "--src-dir", metavar="SRC_DIR", required=True, help="source directory")
    parser.add_argument("-o", "--obj-dir", metavar="OBJ_DIR", required=True, help="build directory")
    parser.add_argument("-t", "--build-type",
                        metavar="BUILD_TYPE", required=True, help="build type, debug or release")

    args = parser.parse_args()

    print("args:{}".format(args))

    return do_configure(args)

if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)

