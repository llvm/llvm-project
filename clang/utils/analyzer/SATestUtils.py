import os
import sys

from subprocess import CalledProcessError, check_call
from typing import List, IO, Optional


def which(command: str, paths: Optional[str] = None) -> Optional[str]:
    """which(command, [paths]) - Look up the given command in the paths string
    (or the PATH environment variable, if unspecified)."""

    if paths is None:
        paths = os.environ.get('PATH', '')

    # Check for absolute match first.
    if os.path.exists(command):
        return command

    # Would be nice if Python had a lib function for this.
    if not paths:
        paths = os.defpath

    # Get suffixes to search.
    # On Cygwin, 'PATHEXT' may exist but it should not be used.
    if os.pathsep == ';':
        pathext = os.environ.get('PATHEXT', '').split(';')
    else:
        pathext = ['']

    # Search the paths...
    for path in paths.split(os.pathsep):
        for ext in pathext:
            p = os.path.join(path, command + ext)
            if os.path.exists(p):
                return p

    return None


def has_no_extension(file_name: str) -> bool:
    root, ext = os.path.splitext(file_name)
    return ext == ""


def is_valid_single_input_file(file_name: str) -> bool:
    root, ext = os.path.splitext(file_name)
    return ext in (".i", ".ii", ".c", ".cpp", ".m", "")


def run_script(script_path: str, build_log_file: IO, cwd: str,
               out=sys.stdout, err=sys.stderr, verbose: int = 0):
    """
    Run the provided script if it exists.
    """
    if os.path.exists(script_path):
        try:
            if verbose == 1:
                out.write(f"  Executing: {script_path}\n")

            check_call(f"chmod +x '{script_path}'", cwd=cwd,
                       stderr=build_log_file,
                       stdout=build_log_file,
                       shell=True)

            check_call(f"'{script_path}'", cwd=cwd,
                       stderr=build_log_file,
                       stdout=build_log_file,
                       shell=True)

        except CalledProcessError:
            err.write(f"Error: Running {script_path} failed. "
                      f"See {build_log_file.name} for details.\n")
            sys.exit(-1)


def is_comment_csv_line(entries: List[str]) -> bool:
    """
    Treat CSV lines starting with a '#' as a comment.
    """
    return len(entries) > 0 and entries[0].startswith("#")
