import os
from subprocess import check_call
import sys


Verbose = 1


def which(command, paths=None):
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


def hasNoExtension(FileName):
    (Root, Ext) = os.path.splitext(FileName)
    return (Ext == "")


def isValidSingleInputFile(FileName):
    (Root, Ext) = os.path.splitext(FileName)
    return Ext in (".i", ".ii", ".c", ".cpp", ".m", "")


def runScript(ScriptPath, PBuildLogFile, Cwd, Stdout=sys.stdout,
              Stderr=sys.stderr):
    """
    Run the provided script if it exists.
    """
    if os.path.exists(ScriptPath):
        try:
            if Verbose == 1:
                Stdout.write("  Executing: %s\n" % (ScriptPath,))
            check_call("chmod +x '%s'" % ScriptPath, cwd=Cwd,
                       stderr=PBuildLogFile,
                       stdout=PBuildLogFile,
                       shell=True)
            check_call("'%s'" % ScriptPath, cwd=Cwd,
                       stderr=PBuildLogFile,
                       stdout=PBuildLogFile,
                       shell=True)
        except:
            Stderr.write("Error: Running %s failed. See %s for details.\n" % (
                         ScriptPath, PBuildLogFile.name))
            sys.exit(-1)


def isCommentCSVLine(Entries):
    """
    Treat CSV lines starting with a '#' as a comment.
    """
    return len(Entries) > 0 and Entries[0].startswith("#")
