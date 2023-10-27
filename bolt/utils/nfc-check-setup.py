#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
import textwrap


def get_git_ref_or_rev(dir: str) -> str:
    # Run 'git symbolic-ref -q --short HEAD || git rev-parse --short HEAD'
    cmd_ref = "git symbolic-ref -q --short HEAD"
    ref = subprocess.run(
        shlex.split(cmd_ref), cwd=dir, text=True, stdout=subprocess.PIPE
    )
    if not ref.returncode:
        return ref.stdout.strip()
    cmd_rev = "git rev-parse --short HEAD"
    return subprocess.check_output(shlex.split(cmd_rev), cwd=dir, text=True).strip()


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
            This script builds two versions of BOLT (with the current and
            previous revision) and sets up symlink for llvm-bolt-wrapper.
            Passes the options through to llvm-bolt-wrapper.
            """
        )
    )
    parser.add_argument(
        "build_dir",
        nargs="?",
        default=os.getcwd(),
        help="Path to BOLT build directory, default is current " "directory",
    )
    parser.add_argument(
        "--switch-back",
        default=False,
        action="store_true",
        help="Checkout back to the starting revision",
    )
    parser.add_argument(
        "--cmp-rev",
        default="HEAD^",
        help="Revision to checkout to compare vs HEAD",
    )
    args, wrapper_args = parser.parse_known_args()
    bolt_path = f"{args.build_dir}/bin/llvm-bolt"

    source_dir = None
    # find the repo directory
    with open(f"{args.build_dir}/CMakeCache.txt") as f:
        for line in f:
            m = re.match(r"LLVM_SOURCE_DIR:STATIC=(.*)", line)
            if m:
                source_dir = m.groups()[0]
    if not source_dir:
        sys.exit("Source directory is not found")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_path = f"{script_dir}/llvm-bolt-wrapper.py"
    # build the current commit
    subprocess.run(
        shlex.split("cmake --build . --target llvm-bolt"), cwd=args.build_dir
    )
    # rename llvm-bolt
    os.replace(bolt_path, f"{bolt_path}.new")
    # memorize the old hash for logging
    old_ref = get_git_ref_or_rev(source_dir)

    # determine whether a stash is needed
    stash = subprocess.run(
        shlex.split("git status --porcelain"),
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    if stash:
        # save local changes before checkout
        subprocess.run(shlex.split("git stash push -u"), cwd=source_dir)
    # check out the previous/cmp commit
    subprocess.run(shlex.split(f"git checkout -f {args.cmp_rev}"), cwd=source_dir)
    # get the parent commit hash for logging
    new_ref = get_git_ref_or_rev(source_dir)
    # build the previous commit
    subprocess.run(
        shlex.split("cmake --build . --target llvm-bolt"), cwd=args.build_dir
    )
    # rename llvm-bolt
    os.replace(bolt_path, f"{bolt_path}.old")
    # set up llvm-bolt-wrapper.ini
    ini = subprocess.check_output(
        shlex.split(f"{wrapper_path} {bolt_path}.old {bolt_path}.new") + wrapper_args,
        text=True,
    )
    with open(f"{args.build_dir}/bin/llvm-bolt-wrapper.ini", "w") as f:
        f.write(ini)
    # symlink llvm-bolt-wrapper
    os.symlink(wrapper_path, bolt_path)
    if args.switch_back:
        if stash:
            subprocess.run(shlex.split("git stash pop"), cwd=source_dir)
        subprocess.run(shlex.split(f"git checkout {old_ref}"), cwd=source_dir)
    else:
        print(
            f"The repository {source_dir} has been switched from {old_ref} "
            f"to {new_ref}. Local changes were stashed. Switch back using\n\t"
            f"git checkout {old_ref}\n"
        )
    print(
        f"Build directory {args.build_dir} is ready to run BOLT tests, e.g.\n"
        "\tbin/llvm-lit -sv tools/bolt/test\nor\n"
        "\tbin/llvm-lit -sv tools/bolttests"
    )


if __name__ == "__main__":
    main()
