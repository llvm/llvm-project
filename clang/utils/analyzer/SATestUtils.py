import os
from subprocess import check_output, check_call
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


class flushfile(object):
    """
    Wrapper to flush the output after every print statement.
    """
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()


def hasNoExtension(FileName):
    (Root, Ext) = os.path.splitext(FileName)
    return (Ext == "")


def isValidSingleInputFile(FileName):
    (Root, Ext) = os.path.splitext(FileName)
    return Ext in (".i", ".ii", ".c", ".cpp", ".m", "")


def getSDKPath(SDKName):
    """
    Get the path to the SDK for the given SDK name. Returns None if
    the path cannot be determined.
    """
    if which("xcrun") is None:
        return None

    Cmd = "xcrun --sdk " + SDKName + " --show-sdk-path"
    return check_output(Cmd, shell=True).rstrip()


def runScript(ScriptPath, PBuildLogFile, Cwd):
    """
    Run the provided script if it exists.
    """
    if os.path.exists(ScriptPath):
        try:
            if Verbose == 1:
                print "  Executing: %s" % (ScriptPath,)
            check_call("chmod +x '%s'" % ScriptPath, cwd=Cwd,
                       stderr=PBuildLogFile,
                       stdout=PBuildLogFile,
                       shell=True)
            check_call("'%s'" % ScriptPath, cwd=Cwd,
                       stderr=PBuildLogFile,
                       stdout=PBuildLogFile,
                       shell=True)
        except:
            print "Error: Running %s failed. See %s for details." % (
                  ScriptPath, PBuildLogFile.name)
            sys.exit(-1)


class Discarder(object):
    """
    Auxiliary object to discard stdout.
    """
    def write(self, text):
        pass  # do nothing


def isCommentCSVLine(Entries):
    """
    Treat CSV lines starting with a '#' as a comment.
    """
    return len(Entries) > 0 and Entries[0].startswith("#")
