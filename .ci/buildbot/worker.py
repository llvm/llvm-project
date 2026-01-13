# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for ScriptedBuilder Buildbot worker scripts"""

import argparse
import filecmp
import os
import stat
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import traceback
import platform
import multiprocessing
from contextlib import contextmanager

_SHQUOTE_WINDOWS_ESCAPEDCHARS = re.compile(r'(["\\])')
_SHQUOTE_WINDOWS_QUOTEDCHARS = re.compile("[ \t\n]")


def _shquote_windows(txt):
    """shlex.quote for Windows cmd.exe"""
    txt = txt.replace("%", "%%")
    quoted = re.sub(_SHQUOTE_WINDOWS_ESCAPEDCHARS, r"\\\1", txt)
    if len(quoted) == len(txt) and not _SHQUOTE_WINDOWS_QUOTEDCHARS.search(txt):
        return txt
    else:
        return '"' + quoted + '"'


def shjoin(args):
    """Convert a list of shell arguments to an appropriately quoted string."""
    if os.name in set(("nt", "os2", "ce")):
        return " ".join(map(_shquote_windows, args))
    else:
        return shlex.join(args)


def report(msg):
    """
    Emit a message to the build log. Appears in red font. Lines surrounded
    by @@@ may be interpreted as meta-instructions.
    """
    print(msg, file=sys.stderr, flush=True)


def report_prog_version(name, cmd):
    try:
        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
        outlines = p.stdout.strip().splitlines()
        report_list(name, outlines[0])
    except BaseException:
        pass


def report_list(category, *items):
    items = list(items)
    filtered = []

    while items:
        item = items.pop()
        match item:
            case tuple() | list():
                items += item
                continue
            case None:
                continue
            case _:
                item = str(item).strip()
        if not item:
            continue
        if item in filtered:
            continue
        filtered.append(item)
    category += ":"
    report(f"{category:<9}{', '.join(reversed( filtered))}")


def report_platform():
    report_list(
        "CPU",
        platform.machine(),
        platform.architecture()[0],
        platform.processor(),
        f"{multiprocessing.cpu_count()} native threads",
    )
    try:
        releaseinfo = platform.freedesktop_os_release()
    except BaseException:
        releaseinfo = dict()
    report_list(
        "OS",
        platform.system(),
        platform.architecture()[1],
        platform.platform(),
        releaseinfo.get("PRETTY_NAME"),
    )
    report_list("Python", platform.python_implementation(), platform.python_version())

    report_prog_version("CMake", ["cmake", "--version"])
    report_prog_version("Ninja", ["ninja", "--version"])
    report_prog_version("Sphinx", ["sphinx-build", "--version"])
    report_prog_version("Doxygen", ["doxygen", "--version"])

    report_prog_version("gcc", ["gcc", "--version"])
    report_prog_version("ld", ["ld", "--version"])

    report_prog_version("LLVM", ["llvm-config", "--version"])
    report_prog_version("Clang", ["clang", "--version"])
    report_prog_version("LLD", ["ld.lld", "--version"])


def run_command(cmd, shell=False, **kwargs):
    """
    Report which command is being run, then execute it using
    subprocess.check_call.
    """
    report(f"Running: {cmd if shell else shjoin(cmd)}")
    sys.stderr.flush()
    subprocess.check_call(cmd, shell=shell, **kwargs)


def _remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal."""
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    func(path)


def rmtree(path):
    """Remove directory path and all its subdirectories. Includes a workaround
    for Windows where shutil.rmtree errors on read-only files.

    Taken from official Python docs
    https://docs.python.org/3/library/shutil.html#rmtree-example
    """
    shutil.rmtree(path, onexc=_remove_readonly)


def try_delete(path):
    """
    Delete the file or directory;
    if not successful, print a warning but continue
    """
    try:
        os.unlink(path)
    except Exception:
        try:
            _remove_readonly(os.unlink, path, _)
        except Exception:
            try:
                rmtree(path)
            except Exception as e:
                print(f"Warning: Could not delete {path}: {e}")


def checkout(giturl, sourcepath):
    """
    Use git to checkout the remote repository giturl at local directory
    sourcepath.

    If the repository already exists, clear all local changes and check out the
    latest main branch.
    """
    if not os.path.exists(sourcepath):
        run_command(["git", "clone", giturl, sourcepath])

    # Reset repository state no matter what there was before
    run_command(["git", "-C", sourcepath, "stash", "--all"])
    run_command(["git", "-C", sourcepath, "stash", "clear"])

    # Fetch and checkout the newest
    run_command(["git", "-C", sourcepath, "fetch", "origin"])
    run_command(["git", "-C", sourcepath, "checkout", "origin/main", "--detach"])


@contextmanager
def step(step_name, halt_on_fail=False):
    """Report a new build step being started.

    Use like this::
        with step("greet-step"):
            report("Hello World!")
    """
    # Barrier to separate stdio output for the the previous step
    sys.stderr.flush()
    sys.stdout.flush()

    report(f"@@@BUILD_STEP {step_name}@@@")
    if halt_on_fail:
        report("@@@HALT_ON_FAILURE@@@")
    try:
        yield
    except Exception as e:
        if isinstance(e, subprocess.CalledProcessError):
            report(f"{shjoin(e.cmd)} exited with return code {e.returncode}.")
            report("@@@STEP_FAILURE@@@")
        else:
            traceback.print_exc()
            report("@@@STEP_EXCEPTION@@@")
        if halt_on_fail:
            # Do not continue with the next steps, but allow except/finally
            # blocks to execute
            raise e


class Worker:
    """Helper class to keep context in a worker.run() environment"""

    def __init__(self, args, clean, clobber, workdir, jobs, cachefile, llvmsrcroot):
        self.args = args
        self.clean = clean
        self.clobber = clobber
        self.workdir = workdir
        self.jobs = jobs
        self.cachefile = cachefile
        self.llvmsrcroot = llvmsrcroot

    def in_llvmsrc(self, path):
        """
        Convert a path in the llvm-project source checkout to an absolute path
        """
        return os.path.join(self.llvmsrcroot, path)

    def in_workdir(self, path):
        """Convert a path in the workdir to an absolute path"""
        return os.path.join(self.workdir, path)

    def run_ninja(
        self, targets: list = [], *, builddir, ccache_stats: bool = False, **kwargs
    ):
        """
        Run ninja in builddir. If self.jobs is set, automatically adds a
        -j option to set the number of parallel jobs.

        Parameters
        ----------
        targets : list
            List of build targets; build the default target 'all' if list is
            empty
        builddir
            Directory of the build.ninja file
        ccache_stats : bool
            If true, also emit ccache statistics when finishing the build
        """
        cmd = ["ninja"]
        if builddir is not None:
            cmd += ["-C", builddir]
        cmd += targets
        if self.jobs:
            cmd.append(f"-j{self.jobs}")
        if ccache_stats:
            run_command(["ccache", "-z"])
            try:
                run_command(cmd, **kwargs)
            finally:
                # TODO: Pipe to stderr to separate from build log itself
                run_command(["ccache", "-sv"])
        else:
            run_command(cmd, **kwargs)

    @contextmanager
    def step(self, step_name, halt_on_fail=False):
        """Convenience wrapper for step()"""
        with step(step_name, halt_on_fail=halt_on_fail) as s:
            yield s

    def report(self, msg):
        """Convenience wrapper for report()"""
        report(msg)

    def run_command(self, *args, **kwargs):
        """Convenience wrapper for run_command()"""
        return run_command(*args, **kwargs)

    def rmtree(self, *args, **kwargs):
        """Convenience wrapper for rmtree()"""
        return rmtree(*args, *kwargs)

    def checkout(self, giturl, sourcepath):
        """Convenience wrapper for checkout()"""
        return checkout(giturl, sourcepath)


def convert_bool(v):
    """Convert input to bool type

    Use to convert the value of bool environment variables. Specifically, the
    buildbot master sets 'false' to build properties, which by default Python
    would interpret as true-ish.
    """
    match v:
        case None:
            return False
        case bool(b):
            return b
        case str(s):
            return not s.strip().upper() in ["", "0", "N", "NO", "FALSE", "OFF"]
        case _:
            return bool(v)


def relative_if_possible(path, relative_to):
    """
    Like os.path.relpath, but does not fail if path is not a parent of
    relative_to; keeps the original path in that case
    """
    path = os.path.normpath(path)
    if not os.path.isabs(path):
        # Path is already relative (assumed to relative_to)
        return path
    try:
        result = os.path.relpath(path, start=relative_to)
        return result if result else path
    except ValueError:
        return path


@contextmanager
def run(
    scriptpath,
    llvmsrcroot,
    parser=None,
    cachefile=None,
    clobberpaths=[],
    workerjobs=None,
    incremental=None,
):
    """
    Runs the boilerplate for a ScriptedBuilder buildbot. It is not necessary to
    use this function (one can also call run_command() etc. directly), but
    allows for some more flexibility and safety checks. Arguments passed to this
    function represent the worker configuration.

    We use the term 'clean' for resetting the worker to an empty state. This
    involves deleting ${prefix}/llvm.src as well as ${prefix}/build.
    The term 'clobber' means deleting build artifacts, but not already
    downloaded git repositories. Build artifacts include build- and
    install-directories. Changes in the llvm.src directory will
    either be force-reset by the buildbot's 'checkout' step anyway,
    or -- in case of local invocation -- represents the source the user wants
    to reproduce without being tied to a specific commit. In either case the
    source directories should not be touched. We consider 'clean' to comprise
    'clobber'. llvm-zorg also uses the term 'clean_obj' instead of 'clobber'.
    By default, we will always clobber to get the same starting point at every
    build. If incremental=True or the --incremental command line option is used,
    the starting point is the previous build.

    A buildbot worker will invoke this script using this directory structure,
    where ${prefix} is a dedicated directory for this builder:
        ${prefix}/llvm.src      # Checkout location for the llvm-source
        ${prefix}/build         # cwd when launching the build script

    The build script is called with --workdir=. parameter, i.e. the build
    artifacts are written into ${prefix}/build. When cleaning, the worker (NOT
    the build script) will delete ${prefix}/llvm.src; Deleting any contents of
    ${prefix}/build is to be done by the builder script, e.g. by this function.
    The builder script can choose to not delete the complete workdir, e.g.
    additional source checkouts such as the llvm-test-suite.

    The buildbot master will set the 'clean' build property and the environment
    variable BUILDBOT_CLEAN when in the GUI the option "Clean source code and
    build directory" is checked by the user. The 'clean_obj' build property and
    the BUILDBOT_CLEAN_OBJ environment variable will be set when either the
    "Clean build directory" GUI option is set, or the master detects a change
    to a CMakeLists.txt or *.cmake file.

    Parameters
    ----------
    scriptpath
        Pass __file__ from the main builder script.
    llvmsrcroot
        Absolute path to the llvm-project source checkout. Since the builder
        script is supposed to be a part of llvm-project itself, the builder
        script can compute it from __file__.
    parser
        Use this argparse.ArgumentParser instead of creating a new one. Allows
        adding additional command line switches in addition to the pre-defined
        ones. Build scripts are encouraged to apply the pre-defined switches.
    cachefile
        Path (relative to llvmsrcroot) of the CMake cache file to
        use. `None` indicates that the script does not use a cache file. Can be
        overridden using --cachefile.
    clobberpaths
        Directories relative to workdir that need to be deleted if the build
        configuration changes (due to changes of CMakeLists.txt or changes of
        configuration parameters). Typically, only source checkouts are not
        deleted.
    workerjobs
        Default number of build and test jobs; If set, expected to be the number
        of jobs of the actual buildbot worker that executes this script. Can be
        overridden using the --jobs parameter so in case someone needs to
        reproduce this build, they can adjust the number of jobs for the
        reproducer platform. Alternatively, the worker can set the
        BUILDBOT_JOBS environment variable or keep ninja/llvm-lit defaults.
    incremental
        Only clobber the build artifacts when the build configuration changes.
        Can be overridden using --incremental.
    """

    scriptpath = os.path.abspath(scriptpath)
    llvmsrcroot = os.path.abspath(llvmsrcroot)
    stem = pathlib.Path(scriptpath).stem
    workdir_default = f"{stem}.workdir"

    jobs_default = None
    if jobs_env := os.environ.get("BUILDBOT_JOBS"):
        jobs_default = int(jobs_env)
    if not jobs_default:
        jobs_default = workerjobs
    if not jobs_default:
        jobs_default = None

    incremental_default = None if incremental else False

    parser = parser or argparse.ArgumentParser(
        allow_abbrev=True,
        description="When executed without arguments, builds the worker's "
        f"LLVM build configuration in {os.path.abspath(workdir_default)}. "
        "Some build configuration parameters can be altered using the "
        "following switches:",
    )
    parser.add_argument(
        "--workdir",
        default=workdir_default,
        help="Use this dir (relative to cwd) as workdir to write the build "
        "artifacts into; --workdir=. uses the current directory.\nWarning: The "
        "content of this directory may be deleted",
    )
    if cachefile is not None:
        parser.add_argument(
            "--cachefile",
            default=relative_if_possible(cachefile, llvmsrcroot),
            help="File containing the initial values for the CMakeCache.txt "
            "for the llvm build.",
        )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=convert_bool(os.environ.get("BUILDBOT_CLEAN")),
        help="Delete the entire workdir before starting the build, including "
        "source directories",
    )
    parser.add_argument(
        "--incremental",
        action=argparse.BooleanOptionalAction,
        default=incremental_default,
        help="Keep previous build artifacts when starting the build",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=jobs_default,
        help="Number of build- and test-jobs",
    )
    args = parser.parse_args()

    workdir = os.path.abspath(args.workdir)
    incremental = args.incremental
    clean = args.clean
    if cachefile is not None:
        cachefile = os.path.join(llvmsrcroot, args.cachefile)
        if not os.path.isfile(cachefile):
            raise Exception(f"--cachefile={cachefile} does not exist")

    prevcachepath = os.path.join(workdir, "prevcache.cmake")
    prevscriptpath = os.path.join(workdir, "prevscript.py")

    if clean:
        # Clean implies clobber
        clobber = False
    elif incremental is None:
        # Automatically determine whether to clobber
        def has_config_change():
            # Has the master scheduler determined a CMakeLists.txt has changed?
            if convert_bool(os.environ.get("BUILDBOT_CLOBBER")):
                return True
            if convert_bool(os.environ.get("BUILDBOT_CLEAN_OBJ")):
                return True

            # Has the build script changed?
            if not os.path.isfile(prevscriptpath):
                return True
            if not filecmp.cmp(scriptpath, prevscriptpath, shallow=False):
                return True

            # Has the cache file (if any) changed?
            if cachefile:
                if not os.path.isfile(prevcachepath):
                    return True
                if not os.path.isfile(cachefile):
                    return True
                if not filecmp.cmp(cachefile, prevcachepath, shallow=False):
                    return True

            return False

        clobber = has_config_change()
    else:
        # Adhere to explicitly set incremental option
        clobber = not incremental

    # Safety check
    parentdir = os.path.dirname(scriptpath)
    while True:
        if os.path.exists(workdir) and os.path.samefile(parentdir, workdir):
            raise Exception(
                f"Cannot use {args.workdir} as workdir; it contains the source "
                "itself in '{parentdir}'"
            )
        newparentdir = os.path.dirname(parentdir)
        if newparentdir == parentdir:
            break
        parentdir = newparentdir

    w = Worker(
        args,
        clean=clean,
        clobber=clobber,
        workdir=workdir,
        jobs=args.jobs,
        cachefile=cachefile,
        llvmsrcroot=llvmsrcroot,
    )

    with step("platform-info"):
        report_platform()

    # Ensure that the cwd is not the directory we are going to delete. This
    # would not work e.g. under Windows. We will chdir to workdir in the next
    # step anyway.
    os.chdir("/")

    if clean:
        if os.path.exists(workdir):
            print("Deleting previous build state including sources", file=sys.stderr)

        with w.step(f"clean"):
            if os.path.exists(workdir):
                # Do not delete the directory itself, just the contents; it might be
                # a symlink to somewhere else
                for d in os.listdir(workdir):
                    try_delete(os.path.join(workdir, d))
    elif clobber:
        # Warn user if deleting anything
        for p in clobberpaths:
            if os.path.exists(os.path.join(workdir, p)):
                print(
                    "Deleting previous build artifacts; use --incremental to keep",
                    file=sys.stderr,
                )
                break

        with w.step(f"clobber"):
            for d in clobberpaths:
                try_delete(os.path.join(workdir, d))
            try_delete(prevscriptpath)
            try_delete(prevcachepath)

    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    # Remember used script and cachefile to detect changes
    shutil.copy(scriptpath, prevscriptpath)
    if cachefile:
        shutil.copy(cachefile, prevcachepath)

    os.environ["NINJA_STATUS"] = "[%p/%es :: %u->%r->%f (of %t)] "

    yield w
