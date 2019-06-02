import argparse
import os
import subprocess
import sys

FILE_EXTENSIONS = [".h", ".hpp", ".c", ".cc", ".cpp"]

def do_clang_tidy(args):
    ret = False

    merge_base_cmd = ["git", "merge-base", "origin/{}".format(args.base_branch), args.branch]
    print(merge_base_cmd)
    base_commit = subprocess.check_output(merge_base_cmd, cwd=args.src_dir)
    base_commit = base_commit.rstrip()

    changed_files = os.path.join(args.builder_dir, "clang_tidy_changed_files.txt")
    if os.path.isfile(changed_files):
        os.remove(changed_files)

    diff_cmd = ["git", "--no-pager", "diff", base_commit, args.branch, "--name-only"]
    print(diff_cmd)
    with open(changed_files, 'w') as f:
        subprocess.check_call(merge_base_cmd, cwd=args.src_dir, stdout=f, stderr=subprocess.STDOUT)

    if os.path.isfile(changed_files):
        clang_tidy_binary = os.path.join(args.obj_dir, "bin", "clang-tidy")
        if os.path.isfile(clang_tidy_binary):
            with open(changed_files, 'r') as f:
                for line in f:
                    filename, file_extension = os.path.splitext(line)
                    if file_extension.lower() in FILE_EXTENSIONS:
                        clang_tidy_cmd = [clang_tidy_binary, line]
                        print(clang_tidy_cmd)
                        subprocess.run(clang_tidy_binary, cwd=args.src_dir)
            ret = True
        else:
            print("no such file:{}".format(clang_tidy_binary))
            # TODO report error when clang-tidy is ready
            print("clang-tidy is not get built, always return success for now.")
            ret = True

    return ret

def main():
    parser = argparse.ArgumentParser(prog="clang_tidy.py",
                                     description="script to do clang_tidy",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--build-number", metavar="BUILD_NUM", help="build number")
    parser.add_argument("-b", "--branch", metavar="BRANCH", required=True, help="pull request branch")
    parser.add_argument("-d", "--base-branch", metavar="BASE_BRANCH", required=True,
                        help="pull request base branch")
    parser.add_argument("-r", "--pr-number", metavar="PR_NUM", help="pull request number")
    parser.add_argument("-w", "--builder-dir", metavar="BUILDER_DIR", required=True,
                        help="builder directory, which is the directory contains source and build directories")
    parser.add_argument("-s", "--src-dir", metavar="SRC_DIR", required=True, help="source directory")
    parser.add_argument("-o", "--obj-dir", metavar="OBJ_DIR", required=True, help="build directory")

    args = parser.parse_args()

    print("args:{}".format(args))

    return do_clang_tidy(args)

if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)

