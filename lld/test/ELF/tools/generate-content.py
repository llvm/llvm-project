#!/usr/bin/env python
"""Given a test file with the following content:

    .ifdef GEN
    #--- a.cc
    int va;
    #--- gen
    clang -S -g a.cc -o -
    .endif
    # content

The script will replace the content after 'endif' with the stdout of 'gen'.
The extra files are generated in a temporary directory with split-file.

Example:
PATH=/path/to/clang_build/bin:$PATH tools/generate-debug-names.py debug-names-*.s Inputs/debug-names*.s
"""
import contextlib, os, subprocess, sys, tempfile


@contextlib.contextmanager
def cd(dir):
    cwd = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(cwd)


def process(path):
    split_file_input = []
    prolog = []
    is_cc = False
    is_prolog = True
    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip()
            if is_prolog:
                prolog.append(line)
            if line.startswith(".endif"):
                is_cc = is_prolog = False
            if is_cc:
                split_file_input.append(line)
            if line.startswith(".ifdef GEN"):
                is_cc = True

    if not split_file_input:
        print("no .ifdef GEN, bail out", file=sys.stderr)
        return
    with tempfile.TemporaryDirectory() as dir:
        sub = subprocess.run(
            ["split-file", "-", dir],
            input="\n".join(split_file_input).encode(),
            capture_output=True,
        )
        if sub.returncode != 0:
            sys.stderr.write(f"split-file failed\n{sub.stderr.decode()}")
            return
        with cd(dir):
            if not os.path.exists("gen"):
                print("'gen' not found", file=sys.stderr)
                return

            env = dict(
                os.environ, CCC_OVERRIDE_OPTIONS="+-fno-ident", PWD="/proc/self/cwd"
            )
            sub = subprocess.run(["zsh", "gen"], capture_output=True, env=env)
            if sub.returncode != 0:
                sys.stderr.write(f"'gen' failed\n{sub.stderr.decode()}")
                return
            content = sub.stdout.decode()

    with open(path, "w") as f:
        # Print lines up to '.endif'.
        print("\n".join(prolog), file=f)
        # Then print the stdout of 'gen'.
        f.write(content)


for path in sys.argv[1:]:
    process(path)
