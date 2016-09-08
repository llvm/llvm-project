""" Post process SWIG Bridge wrapper code Python script for Windows/LINUX/OSX platform

    --------------------------------------------------------------------------
    File:           finishSwigWrapperClasses.py

    Overview:       Python script(s) to finish off the SWIG Python C++ Script
                    Bridge wrapper code on the Windows/LINUX/OSX platform.
                    The Python scripts are equivalent to the shell script (.sh)
                    files.
                    We use SWIG to create a C++ file containing the appropriate
                    wrapper classes and functions for each scripting language,
                    before liblldb is built (thus the C++ file can be compiled
                    into liblldb.  In some cases, additional work may need to be
                    done after liblldb has been compiled, to make the scripting
                    language stuff fully functional.  Any such post-processing
                    is handled through the Python scripts called here.

    Gotchas:        None.

    Copyright:      None.
    --------------------------------------------------------------------------

"""

# Python modules:
import sys      # Provide argument parsing
import os       # Provide directory and file handling

# Third party modules:

# In-house modules:
import utilsArgsParse   # Parse and validate this script's input arguments
import utilsOsType      # Determine the OS type this script is running on
import utilsDebug       # Debug Python scripts

# Instantiations:
# True = Turn on script function tracing, False = off.
gbDbgVerbose = False
gbDbgFlag = False              # Global debug mode flag, set by input parameter
# --dbgFlag. True = operate in debug mode.
# True = yes called from makefile system, False = not.
gbMakeFileFlag = False

# User facing text:
strMsgErrorNoMain = "Program called by another Python script not allowed"
strExitMsgSuccess = "Program successful"
strExitMsgError = "Program error: "
strParameter = "Parameter: "
strMsgErrorOsTypeUnknown = "Unable to determine OS type"
strScriptDirNotFound = "Unable to locate the script directory \'/script\'"
strScriptLangsFound = "Found the following script languages:"
strPostProcessError = "Executing \'%s\' post process script failed: "
strScriptNotFound = "Unable to locate the post process script file \'%s\' in \'%s\'"
strScriptLangFound = "Found \'%s\' build script."
strScriptLangsFound = "Found the following script languages:"
strExecuteMsg = "Executing \'%s\' build script..."
strExecuteError = "Executing \'%s\' build script failed: "
strHelpInfo = "\
Python script(s) to finish off the SWIG Python C++ Script \n\
Bridge wrapper code on the Windows/LINUX/OSX platform.  The Python \n\
scripts are equivalent to the shell script (.sh) files \n\
run on others platforms.\n\
Args:   -h              (optional) Print help information on this program.\n\
    -d              (optional) Determines whether or not this script\n\
                    outputs additional information when running.\n\
    -m              (optional) Specify called from Makefile system.\n\
    --srcRoot=          The root of the lldb source tree.\n\
    --targetDir=            Where the lldb framework/shared library gets put.\n\
    --cfgBldDir=            (optional) Where the build-swig-Python-LLDB.py program \n\
                    will put the lldb.py file it generated from running\n\
                    SWIG.\n\
    --prefix=           (optional) Is the root directory used to determine where\n\
                    third-party modules for scripting languages should\n\
                    be installed. Where non-Darwin systems want to put\n\
                    the .py and .so files so that Python can find them\n\
                    automatically. Python install directory.\n\
    --cmakeBuildConfiguration=  (optional) Is the build configuration(Debug, Release, RelWithDebugInfo)\n\
                    used to determine where the bin and lib directories are \n\
                    created for a Windows build.\n\
    --argsFile=         The args are read from a file instead of the\n\
                    command line. Other command line args are ignored.\n\
\n\
Usage:\n\
    finishSwigWrapperClasses.py --srcRoot=ADirPath --targetDir=ADirPath\n\
    --cfgBldDir=ADirPath --prefix=ADirPath -m -d\n\
\n\
"  # TAG_PROGRAM_HELP_INFO

#++---------------------------------------------------------------------------
# Details:  Exit the program on success. Called on program successfully done
#           its work. Returns a status result to the caller.
# Args:     vnResult    - (R) 0 or greater indicating success.
#           vMsg        - (R) Success message if any to show success to user.
# Returns:  None.
# Throws:   None.
#--


def program_exit_success(vnResult, vMsg):
    strMsg = ""

    if vMsg.__len__() != 0:
        strMsg = "%s: %s (%d)" % (strExitMsgSuccess, vMsg, vnResult)
        print(strMsg)

    sys.exit(vnResult)

#++---------------------------------------------------------------------------
# Details:  Exit the program with error. Called on exit program failed its
#           task. Returns a status result to the caller.
# Args:     vnResult    - (R) A negative number indicating error condition.
#           vMsg        - (R) Error message to show to user.
# Returns:  None.
# Throws:   None.
#--


def program_exit_on_failure(vnResult, vMsg):
    print(("%s%s (%d)" % (strExitMsgError, vMsg, vnResult)))
    sys.exit(vnResult)

#++---------------------------------------------------------------------------
# Details:  Exit the program return a exit result number and print a message.
#           Positive numbers and zero are returned for success other error
#           occurred.
# Args:     vnResult    - (R) A -ve (an error), 0 or +ve number (ok or status).
#           vMsg        - (R) Error message to show to user.
# Returns:  None.
# Throws:   None.
#--


def program_exit(vnResult, vMsg):
    if vnResult >= 0:
        program_exit_success(vnResult, vMsg)
    else:
        program_exit_on_failure(vnResult, vMsg)

#++---------------------------------------------------------------------------
# Details:  Dump input parameters.
# Args:     vDictArgs   - (R) Map of input args to value.
# Returns:  None.
# Throws:   None.
#--


def print_out_input_parameters(vDictArgs):
    for arg, val in list(vDictArgs.items()):
        strEqs = ""
        strQ = ""
        if val.__len__() != 0:
            strEqs = " ="
            strQ = "\""
        print(("%s%s%s %s%s%s\n" % (strParameter, arg, strEqs, strQ, val, strQ)))

#++---------------------------------------------------------------------------
# Details:  Validate the arguments passed to the program. This function exits
#           the program should error with the arguments be found.
# Args:     vArgv   - (R) List of arguments and values.
# Returns:  Int     - 0 = success, -ve = some failure.
#           Dict    - Map of arguments names to argument values
# Throws:   None.
#--


def validate_arguments(vArgv):
    dbg = utilsDebug.CDebugFnVerbose("validate_arguments()")
    strMsg = ""
    dictArgs = {}
    nResult = 0
    strListArgs = "hdm"  # Format "hiox:" = -h -i -o -x <arg>
    listLongArgs = [
        "srcRoot=",
        "targetDir=",
        "cfgBldDir=",
        "prefix=",
        "cmakeBuildConfiguration=",
        "argsFile"]
    dictArgReq = {"-h": "o",          # o = optional, m = mandatory
                  "-d": "o",
                  "-m": "o",
                  "--srcRoot": "m",
                  "--targetDir": "m",
                  "--cfgBldDir": "o",
                  "--prefix": "o",
                  "--cmakeBuildConfiguration": "o",
                  "--argsFile": "o"}

    # Check for mandatory parameters
    nResult, dictArgs, strMsg = utilsArgsParse.parse(vArgv, strListArgs,
                                                     listLongArgs,
                                                     dictArgReq,
                                                     strHelpInfo)
    if nResult < 0:
        program_exit_on_failure(nResult, strMsg)

    # User input -h for help
    if nResult == 1:
        program_exit_success(0, strMsg)

    return (nResult, dictArgs)

#++---------------------------------------------------------------------------
# Details:  Locate post process script language directory and the script within
#           and execute.
# Args:     vStrScriptLang      - (R) Name of the script language to build.
#           vstrFinishFileName  - (R) Prefix file name to build full name.
#           vDictArgs           - (R) Program input parameters.
# Returns:  Int     - 0 = Success, < 0 some error condition.
#           Str     - Error message.
# Throws:   None.
#--


def run_post_process(vStrScriptLang, vstrFinishFileName, vDictArgs):
    dbg = utilsDebug.CDebugFnVerbose("run_post_process()")
    nResult = 0
    strStatusMsg = ""
    strScriptFile = vstrFinishFileName % vStrScriptLang
    strScriptFileDir = os.path.normpath(
        os.path.join(
            vDictArgs["--srcRoot"],
            "scripts",
            vStrScriptLang))
    strScriptFilePath = os.path.join(strScriptFileDir, strScriptFile)

    # Check for the existence of the script file
    strPath = os.path.normcase(strScriptFilePath)
    bOk = os.path.exists(strPath)
    if not bOk:
        strDir = os.path.normcase(strScriptFileDir)
        strStatusMsg = strScriptNotFound % (strScriptFile, strDir)
        return (-9, strStatusMsg)

    if gbDbgFlag:
        print((strScriptLangFound % vStrScriptLang))
        print((strExecuteMsg % vStrScriptLang))

    # Change where Python looks for our modules
    strDir = os.path.normcase(strScriptFileDir)
    sys.path.append(strDir)

    # Execute the specific language script
    dictArgs = vDictArgs  # Remove any args not required before passing on
    strModuleName = strScriptFile[: strScriptFile.__len__() - 3]
    module = __import__(strModuleName)
    nResult, strStatusMsg = module.main(dictArgs)

    # Revert sys path
    sys.path.remove(strDir)

    return (nResult, strStatusMsg)

#++---------------------------------------------------------------------------
# Details:  Step through each script language sub directory supported
#           and execute post processing script for each scripting language,
#           make sure the build script for that language exists.
#           For now the only language we support is Python, but we expect this
#           to change.
# Args:     vDictArgs   - (R) Program input parameters.
# Returns:  Int     - 0 = Success, < 0 some error condition.
#           Str     - Error message.
# Throws:   None.
#--


def run_post_process_for_each_script_supported(vDictArgs):
    dbg = utilsDebug.CDebugFnVerbose(
        "run_post_process_for_each_script_supported()")
    nResult = 0
    strStatusMsg = ""
    strScriptDir = os.path.normpath(
        os.path.join(
            vDictArgs["--srcRoot"],
            "scripts"))
    strFinishFileName = "finishSwig%sLLDB.py"

    # Check for the existence of the scripts folder
    strScriptsDir = os.path.normcase(strScriptDir)
    bOk = os.path.exists(strScriptsDir)
    if not bOk:
        return (-8, strScriptDirNotFound)

    # Look for any script language directories to build for
    listDirs = ["Python"]

    # Iterate script directory find any script language directories
    for scriptLang in listDirs:
        # __pycache__ is a magic directory in Python 3 that holds .pyc files
        if scriptLang != "__pycache__" and scriptLang != "swig_bot_lib":
            dbg.dump_text("Executing language script for \'%s\'" % scriptLang)
            nResult, strStatusMsg = run_post_process(
                scriptLang, strFinishFileName, vDictArgs)
        if nResult < 0:
            break

    if nResult < 0:
        strTmp = strPostProcessError % scriptLang
        strTmp += strStatusMsg
        strStatusMsg = strTmp

    return (nResult, strStatusMsg)

#++---------------------------------------------------------------------------
# Details:  Program's main() with arguments passed in from the command line.
#           Program either exits normally or with error from this function -
#           top most level function.
# Args:     vArgv   - (R) List of arguments and values.
# Returns:  None
# Throws:   None.
#--


def main(vArgv):
    dbg = utilsDebug.CDebugFnVerbose("main()")
    bOk = False
    dictArgs = {}
    nResult = 0
    strMsg = ""

    # The validate arguments fn will exit the program if tests fail
    nResult, dictArgs = validate_arguments(vArgv)

    eOSType = utilsOsType.determine_os_type()
    if eOSType == utilsOsType.EnumOsType.Unknown:
        program_exit(-4, strMsgErrorOsTypeUnknown)

    global gbDbgFlag
    gbDbgFlag = "-d" in dictArgs
    if gbDbgFlag:
        print_out_input_parameters(dictArgs)

    # Check to see if we were called from the Makefile system. If we were, check
    # if the caller wants SWIG to generate a dependency file.
    # Not used in this program, but passed through to the language script file
    # called by this program
    global gbMakeFileFlag
    gbMakeFileFlag = "-m" in dictArgs

    nResult, strMsg = run_post_process_for_each_script_supported(dictArgs)

    program_exit(nResult, strMsg)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# TAG_PROGRAM_HELP_INFO
""" Details: Program main entry point.

    --------------------------------------------------------------------------
    Args:   -h (optional)   Print help information on this program.
            -d (optional)   Determines whether or not this script
                            outputs additional information when running.
            -m (optional)   Specify called from Makefile system. If given locate
                            the LLDBWrapPython.cpp in --srcRoot/source folder
                            else in the --targetDir folder.
            --srcRoot=      The root of the lldb source tree.
            --targetDir=    Where the lldb framework/shared library gets put.
            --cfgBldDir=    Where the buildSwigPythonLLDB.py program will
            (optional)      put the lldb.py file it generated from running
                            SWIG.
            --prefix=       Is the root directory used to determine where
            (optional)      third-party modules for scripting languages should
                            be installed. Where non-Darwin systems want to put
                            the .py and .so files so that Python can find them
                            automatically. Python install directory.
            --cmakeBuildConfiguration=  (optional) Is the build configuration(Debug, Release, RelWithDebugInfo)\n\
                            used to determine where the bin and lib directories are \n\
                            created for a Windows build.\n\
            --argsFile=     The args are read from a file instead of the
                            command line. Other command line args are ignored.
    Usage:
            finishSwigWrapperClasses.py --srcRoot=ADirPath --targetDir=ADirPath
            --cfgBldDir=ADirPath --prefix=ADirPath -m -d

    Results:    0 Success
                -1 Error - invalid parameters passed.
                -2 Error - incorrect number of mandatory parameters passed.

                -4 Error - unable to determine OS type.
                -5 Error - program not run with name of "__main__".
                -8 Error - unable to locate the scripts folder.
                -9 Error - unable to locate the post process language script
                           file.

                -100+    - Error messages from the child language script file.

    --------------------------------------------------------------------------

"""

# Called using "__main__" when not imported i.e. from the command line
if __name__ == "__main__":
    utilsDebug.CDebugFnVerbose.bVerboseOn = gbDbgVerbose
    dbg = utilsDebug.CDebugFnVerbose("__main__")
    main(sys.argv[1:])
else:
    program_exit(-5, strMsgErrorNoMain)
