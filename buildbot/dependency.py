import argparse
import os
import shutil
import subprocess
import sys

def do_dependency(args):
    ret = False

    # hack for incremental build
    if args.pr_number is not None and not args.clean_build:
        if args.branch is None or args.base_branch is None:
            "branch ({}) and base branch ({}) is required for pull request #{}".format(
                args.branch, args.base_branch, args.pr_number)
            return ret
        # fetching the recent state of base branch
        fetch_cmd = ["git", "fetch", "origin", args.base_branch]
        print(fetch_cmd)
        subprocess.check_call(fetch_cmd, cwd=args.src_dir)
        # fetching and checkout PR changes
        fetch_pr_cmd = ["git", "fetch", "-t", "origin", args.branch]
        print(fetch_pr_cmd)
        subprocess.check_call(fetch_pr_cmd, cwd=args.src_dir)
        checkout_cmd = ["git", "checkout", "-B", args.branch]
        print(checkout_cmd)
        subprocess.check_call(checkout_cmd, cwd=args.src_dir)
        # get baseline commit
        merge_base_cmd = ["git", "merge-base", "origin/{}".format(args.base_branch), args.branch]
        print(merge_base_cmd)
        base_commit = subprocess.check_output(merge_base_cmd, cwd=args.src_dir)
        base_commit = base_commit.rstrip() 
        diff_cmd = ["git", "--no-pager", "diff", base_commit, args.branch, "--name-only", "buildbot"]
        print(diff_cmd)
        changed_build_scripts = subprocess.check_output(diff_cmd, cwd=args.src_dir)
        changed_build_scripts = changed_build_scripts.rstrip() if changed_build_scripts is not None else None
        # clean build directory if build scripts have changed
        if len(changed_build_scripts) > 0:
            if os.path.isdir(args.obj_dir):
                shutil.rmtree(args.obj_dir)
                if os.path.isdir(args.obj_dir):
                    print("failed to remove build directory: {}".format(args.obj_dir))
                    return ret
                # let's recover it
                os.makedirs(args.obj_dir)
                if not os.path.isdir(args.obj_dir):
                    print("failed to recover build directory: {}".format(args.obj_dir))
                    return ret

    # fetch OpenCL headers
    ocl_header_dir = os.path.join(args.obj_dir, "OpenCL-Headers")
    if not os.path.isdir(ocl_header_dir):
        clone_cmd = ["git", "clone", "https://github.com/KhronosGroup/OpenCL-Headers", "OpenCL-Headers"]
        subprocess.check_call(clone_cmd, cwd=args.obj_dir)
    else:
        fetch_cmd = ["git", "pull", "--ff", "--ff-only", "origin"]
        subprocess.check_call(fetch_cmd, cwd=ocl_header_dir)

    # fetch and build OpenCL ICD loader
    icd_loader_dir = os.path.join(args.obj_dir, "OpenCL-ICD-Loader")
    if not os.path.isdir(icd_loader_dir):
        clone_cmd = ["git", "clone", "https://github.com/KhronosGroup/OpenCL-ICD-Loader", "OpenCL-ICD-Loader"]
        subprocess.check_call(clone_cmd, cwd=args.obj_dir)
    else:
        fetch_cmd = ["git", "pull", "--ff", "--ff-only", "origin"]
        subprocess.check_call(fetch_cmd, cwd=icd_loader_dir)

    icd_build_dir = os.path.join(icd_loader_dir, "build")
    if os.path.isdir(icd_build_dir):
        shutil.rmtree(icd_build_dir)
    os.makedirs(icd_build_dir)

    cmake_cmd = ["cmake", ".."]
    subprocess.check_call(cmake_cmd, cwd=icd_build_dir)

    make_cmd = ["make", "C_INCLUDE_PATH={}".format(ocl_header_dir)]
    subprocess.check_call(make_cmd, cwd=icd_build_dir)

    ret = True
    return ret

def main():
    parser = argparse.ArgumentParser(prog="dependency.py",
                                     description="script to get and build dependency",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--build-number", metavar="BUILD_NUM", help="build number")
    parser.add_argument("-b", "--branch", metavar="BRANCH", help="pull request branch")
    parser.add_argument("-d", "--base-branch", metavar="BASE_BRANCH", help="pull request base branch")
    parser.add_argument("-r", "--pr-number", metavar="PR_NUM", help="pull request number")
    parser.add_argument("-w", "--builder-dir", metavar="BUILDER_DIR",
                        help="builder directory, which is the directory contains source and build directories")
    parser.add_argument("-s", "--src-dir", metavar="SRC_DIR", help="source directory")
    parser.add_argument("-o", "--obj-dir", metavar="OBJ_DIR", required=True, help="build directory")
    parser.add_argument("-c", "--clean-build", action="store_true", default=False,
                        help="true if the build is clean build which has clobber step")

    args = parser.parse_args()

    print("args:{}".format(args))

    return do_dependency(args)

if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)

