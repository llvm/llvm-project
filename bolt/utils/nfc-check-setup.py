#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
import textwrap

msg_prefix = "\n> NFC-Mode:"

def get_relevant_bolt_changes(dir: str) -> str:
    # Return a list of bolt source changes that are relevant to testing.
    all_changes = subprocess.run(
        shlex.split("git show HEAD --name-only --pretty=''"),
        cwd=dir,
        text=True,
        stdout=subprocess.PIPE,
    )
    keep_bolt = subprocess.run(
        shlex.split("grep '^bolt'"),
        input=all_changes.stdout,
        text=True,
        stdout=subprocess.PIPE,
    )
    keep_relevant = subprocess.run(
        shlex.split(
            "grep -v -e '^bolt/docs' -e '^bolt/utils/docker' -e '^bolt/utils/dot2html'"
        ),
        input=keep_bolt.stdout,
        text=True,
        stdout=subprocess.PIPE,
    )
    return keep_relevant.stdout

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

def switch_back(
    switch_back: bool, stash: bool, source_dir: str, old_ref: str, new_ref: str
):
    # Switch back to the current revision if needed and inform the user of where
    # the HEAD is. Must be called after checking out the previous commit on all
    # exit paths.
    if switch_back:
        print(f"{msg_prefix} Switching back to current revision..")
        if stash:
            subprocess.run(shlex.split("git stash pop"), cwd=source_dir)
        subprocess.run(shlex.split(f"git checkout {old_ref}"), cwd=source_dir)
    else:
        print(
            f"The repository {source_dir} has been switched from {old_ref} "
            f"to {new_ref}. Local changes were stashed. Switch back using\n\t"
            f"git checkout {old_ref}\n"
        )

def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
            This script builds two versions of BOLT:
            llvm-bolt.new, using the current revision, and llvm-bolt.old using
            the previous revision. These can be used to check whether the
            current revision changes BOLT's functional behavior.
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
        "--create-wrapper",
        default=False,
        action="store_true",
        help="Sets up llvm-bolt as a symlink to llvm-bolt-wrapper. Passes the options through to llvm-bolt-wrapper.",
    )
    parser.add_argument(
        "--check-bolt-sources",
        default=False,
        action="store_true",
        help="Create a marker file (.llvm-bolt.changes) if any relevant BOLT sources are modified",
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

    # When creating a wrapper, pass any unknown arguments to it. Otherwise, die.
    args, wrapper_args = parser.parse_known_args()
    if not args.create_wrapper and len(wrapper_args) > 0:
        parser.parse_args()

    # Find the repo directory.
    source_dir = None
    try:
        CMCacheFilename = f"{args.build_dir}/CMakeCache.txt"
        with open(CMCacheFilename) as f:
            for line in f:
                m = re.match(r"LLVM_SOURCE_DIR:STATIC=(.*)", line)
                if m:
                    source_dir = m.groups()[0]
        if not source_dir:
            raise Exception(f"Source directory not found: '{CMCacheFilename}'")
    except Exception as e:
        sys.exit(e)

    # Clean the previous llvm-bolt if it exists.
    bolt_path = f"{args.build_dir}/bin/llvm-bolt"
    if os.path.exists(bolt_path):
        os.remove(bolt_path)

    # Build the current commit.
    print(f"{msg_prefix} Building current revision..")
    subprocess.run(
        shlex.split("cmake --build . --target llvm-bolt"), cwd=args.build_dir
    )

    if not os.path.exists(bolt_path):
        sys.exit(f"Failed to build the current revision: '{bolt_path}'")

    # Rename llvm-bolt and memorize the old hash for logging.
    os.replace(bolt_path, f"{bolt_path}.new")
    old_ref = get_git_ref_or_rev(source_dir)

    if args.check_bolt_sources:
        marker = f"{args.build_dir}/.llvm-bolt.changes"
        if os.path.exists(marker):
            os.remove(marker)
        file_changes = get_relevant_bolt_changes(source_dir)
        # Create a marker file if any relevant BOLT source files changed.
        if len(file_changes) > 0:
            print(f"BOLT source changes were found:\n{file_changes}")
            open(marker, "a").close()

    # Determine whether a stash is needed.
    stash = subprocess.run(
        shlex.split("git status --porcelain"),
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    if stash:
        # Save local changes before checkout.
        subprocess.run(shlex.split("git stash push -u"), cwd=source_dir)

    # Check out the previous/cmp commit and get its commit hash for logging.
    subprocess.run(shlex.split(f"git checkout -f {args.cmp_rev}"), cwd=source_dir)
    new_ref = get_git_ref_or_rev(source_dir)

    # Build the previous commit.
    print(f"{msg_prefix} Building previous revision..")
    subprocess.run(
        shlex.split("cmake --build . --target llvm-bolt"), cwd=args.build_dir
    )

    # Rename llvm-bolt.
    if not os.path.exists(bolt_path):
        print(f"Failed to build the previous revision: '{bolt_path}'")
        switch_back(args.switch_back, stash, source_dir, old_ref, new_ref)
        sys.exit(1)
    os.replace(bolt_path, f"{bolt_path}.old")

    # Symlink llvm-bolt-wrapper
    if args.create_wrapper:
        print(f"{msg_prefix} Creating llvm-bolt wrapper..")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        wrapper_path = f"{script_dir}/llvm-bolt-wrapper.py"
        try:
            # Set up llvm-bolt-wrapper.ini
            ini = subprocess.check_output(
                shlex.split(f"{wrapper_path} {bolt_path}.old {bolt_path}.new")
                + wrapper_args,
                text=True,
            )
            with open(f"{args.build_dir}/bin/llvm-bolt-wrapper.ini", "w") as f:
                f.write(ini)
            os.symlink(wrapper_path, bolt_path)
        except Exception as e:
            print("Failed to create a wrapper:\n" + str(e))
            switch_back(args.switch_back, stash, source_dir, old_ref, new_ref)
            sys.exit(1)

    switch_back(args.switch_back, stash, source_dir, old_ref, new_ref)

    print(
        f"{msg_prefix} Completed!\nBuild directory {args.build_dir} is ready for"
        " NFC-Mode comparison between the two revisions."
    )

    if args.create_wrapper:
        print(
            "Can run BOLT tests using:\n"
            "\tbin/llvm-lit -sv tools/bolt/test\nor\n"
            "\tbin/llvm-lit -sv tools/bolttests"
        )


if __name__ == "__main__":
    main()
