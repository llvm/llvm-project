""" Python SWIG wrapper creation script Windows/LINUX/OSX platform

    --------------------------------------------------------------------------
    File:      buildSwigPython.py

    Overview:  Creates SWIG Python C++ Script Bridge wrapper code. This
               script is called by build-swig-wrapper-classes.py in turn.

    Gotchas:   Python debug complied pythonXX_d.lib is required for SWIG
               to build correct LLDBWrapperPython.cpp in order for Visual
               Studio to compile successfully. The release version of the
               Python lib will not work.
               LLDB (dir) CMakeLists.txt uses windows environmental
               variables $PYTHON_INCLUDE and $PYTHON_LIB to locate
               Python files required for the build.

    Copyright: None.
    --------------------------------------------------------------------------

"""

# Python modules:
import os         # Provide directory and file handling, determine OS information
import sys        # sys.executable
import subprocess # Call external programs
import shutil     # File handling

# Third party modules:

# In-house modules:
import utilsOsType # Determine the OS type this script is running on
import utilsDebug  # Debug Python scripts

# User facing text:
strMsgLldbDisablePythonEnv = "Python build aborted as LLDB_DISABLE_PYTHON \
environmental variable is defined"
strMsgLldbDisableGccEnv = "Python build aborted as GCC_PREPROCESSOR_DEFINITIONS \
environmental variable is defined"
strMsgHdrFiles = "Header files are:"
strMsgIFaceFiles = "SWIG interface files are"
strErrMsgProgFail = "Program failure: "
strMsgFileNewrSwigOpFile = "\'%s\' is newer than \'%s\'\nSWIG file will need to be re-built"
strMsgFileNotExist = "\'%s\' could not be found\nSWIG file will need to be re-built"
strErrMsgOsTypeUnknown = "Unable to determine current OS type"
strMsgNotNeedUpdate = "Everything is up-to-date"
strMsgSwigNeedRebuild = "SWIG needs to be re-run"
strErrMsgSwigParamsMissing = "This script was not passed '--swigExecutable' as required."
strMsgSwigExecute = "SWIG executing the following:\n\'%s'"
strErrMsgSwigExecute = "SWIG failed: %s"
strErrMsgPythonExecute = "Python script '%s' failed: %s"
strMsgSwigNoGenDep = "SWIG ran with no generated dependencies, script exit early"
strMsgSwigGenDep = "SWIG ran and generated dependencies, script exit early, deleted '%s'"
strErrMsgFrameWkPyDirNotExist = "Unable to find the LLDB. Framework directory is '%s'"
strMsgFoundLldbFrameWkDir = "Found '%s'"
strErrMsgModifyPythonLldbPyFileNotFound = "Unable to find '%s' in '%s'"

#++---------------------------------------------------------------------------
# Details: Retrieve the list of hard coded lldb header file names and
#          put in the program's dictArgs map container. Make paths compatible
#          with the current OS.
#          Note this does not necessarily match the content of those
#          directories. The resultant header string is inserted into the
#          dictionary vDictArgs key "--headerFiles".
# Args:    vDictArgs - (RW) Program input parameters.
# Returns: Bool - True = success, False = failure.
# Throws: None.
#--
def get_header_files(vDictArgs):
    dbg = utilsDebug.CDebugFnVerbose("Python script get_header_files()")
    listHeaderFiles = ["/include/lldb/API/lldb.h", # .sh has /include/lldb/lldb.h 21/11/2013
                       "/include/lldb/lldb-defines.h",
                       "/include/lldb/lldb-enumerations.h",
                       "/include/lldb/lldb-forward.h",
                       "/include/lldb/lldb-forward-rtti.h",
                       "/include/lldb/lldb-types.h",
                       "/include/lldb/API/SBAddress.h",
                       "/include/lldb/API/SBAttachInfo.h",
                       "/include/lldb/API/SBBlock.h",
                       "/include/lldb/API/SBBreakpoint.h",
                       "/include/lldb/API/SBBreakpointLocation.h",
                       "/include/lldb/API/SBBroadcaster.h",
                       "/include/lldb/API/SBCommandInterpreter.h",
                       "/include/lldb/API/SBCommandReturnObject.h",
                       "/include/lldb/API/SBCommunication.h",
                       "/include/lldb/API/SBCompileUnit.h",
                       "/include/lldb/API/SBData.h",
                       "/include/lldb/API/SBDebugger.h",
                       "/include/lldb/API/SBError.h",
                       "/include/lldb/API/SBEvent.h",
                       "/include/lldb/API/SBExpressionOptions.h",
                       "/include/lldb/API/SBFileSpec.h",
                       "/include/lldb/API/SBFrame.h",
                       "/include/lldb/API/SBFunction.h",
                       "/include/lldb/API/SBHostOS.h",
                       "/include/lldb/API/SBInputReader.h",
                       "/include/lldb/API/SBInstruction.h",
                       "/include/lldb/API/SBInstructionList.h",
                       "/include/lldb/API/SBLanguageRuntime.h",
                       "/include/lldb/API/SBLaunchInfo.h",
                       "/include/lldb/API/SBLineEntry.h",
                       "/include/lldb/API/SBListener.h",
                       "/include/lldb/API/SBModule.h",
                       "/include/lldb/API/SBModuleSpec.h",
                       "/include/lldb/API/SBProcess.h",
                       "/include/lldb/API/SBSourceManager.h",
                       "/include/lldb/API/SBStream.h",
                       "/include/lldb/API/SBStringList.h",
                       "/include/lldb/API/SBSymbol.h",
                       "/include/lldb/API/SBSymbolContext.h",
                       "/include/lldb/API/SBSymbolContextList.h",
                       "/include/lldb/API/SBTarget.h",
                       "/include/lldb/API/SBThread.h",
                       "/include/lldb/API/SBThreadCollection.h",
                       "/include/lldb/API/SBType.h",
                       "/include/lldb/API/SBTypeCategory.h",
                       "/include/lldb/API/SBTypeFilter.h",
                       "/include/lldb/API/SBTypeFormat.h",
                       "/include/lldb/API/SBTypeNameSpecifier.h",
                       "/include/lldb/API/SBTypeSummary.h",
                       "/include/lldb/API/SBTypeSynthetic.h",
                       "/include/lldb/API/SBValue.h",
                       "/include/lldb/API/SBValueList.h",
                       "/include/lldb/API/SBWatchpoint.h"]
    bDebug = "-d" in vDictArgs
    strRt = vDictArgs["--srcRoot"]
    strRt = os.path.normcase(strRt)

    strHeaderFiles = ""
    for strHdr in listHeaderFiles[0: len(listHeaderFiles)]:
        strHdr = os.path.normcase(strHdr)
        strHeaderFiles += " %s%s" % (strRt, strHdr)

    if bDebug:
        print(strMsgHdrFiles)
        print(strHeaderFiles)

    vDictArgs["--headerFiles"] = strHeaderFiles

    return True

#++---------------------------------------------------------------------------
# Details: Retrieve the list of hard coded lldb SWIG interface file names and
#          put in the program's dictArgs map container. Make paths compatible
#          with the current OS.
#          Note this does not necessarily match the content of those
#          directories. The resultant interface string is inserted into the
#          dictionary vDictArgs key "--ifaceFiles".
# Args:    vDictArgs - (RW) Program input parameters.
# Returns: Bool - True = success, False = failure.
# Throws:  None.
#--
def get_interface_files(vDictArgs):
    dbg = utilsDebug.CDebugFnVerbose("Python script get_interface_files()")
    listIFaceFiles = ["/scripts/interface/SBAddress.i",
                      "/scripts/interface/SBAttachInfo.i",
                      "/scripts/interface/SBBlock.i",
                      "/scripts/interface/SBBreakpoint.i",
                      "/scripts/interface/SBBreakpointLocation.i",
                      "/scripts/interface/SBBroadcaster.i",
                      "/scripts/interface/SBCommandInterpreter.i",
                      "/scripts/interface/SBCommandReturnObject.i",
                      "/scripts/interface/SBCommunication.i",
                      "/scripts/interface/SBCompileUnit.i",
                      "/scripts/interface/SBData.i",
                      "/scripts/interface/SBDebugger.i",
                      "/scripts/interface/SBDeclaration.i",
                      "/scripts/interface/SBError.i",
                      "/scripts/interface/SBEvent.i",
                      "/scripts/interface/SBExpressionOptions.i",
                      "/scripts/interface/SBFileSpec.i",
                      "/scripts/interface/SBFrame.i",
                      "/scripts/interface/SBFunction.i",
                      "/scripts/interface/SBHostOS.i",
                      "/scripts/interface/SBInputReader.i",
                      "/scripts/interface/SBInstruction.i",
                      "/scripts/interface/SBInstructionList.i",
                      "/scripts/interface/SBLanguageRuntime.i",
                      "/scripts/interface/SBLaunchInfo.i",
                      "/scripts/interface/SBLineEntry.i",
                      "/scripts/interface/SBListener.i",
                      "/scripts/interface/SBModule.i",
                      "/scripts/interface/SBModuleSpec.i",
                      "/scripts/interface/SBProcess.i",
                      "/scripts/interface/SBSourceManager.i",
                      "/scripts/interface/SBStream.i",
                      "/scripts/interface/SBStringList.i",
                      "/scripts/interface/SBSymbol.i",
                      "/scripts/interface/SBSymbolContext.i",
                      "/scripts/interface/SBTarget.i",
                      "/scripts/interface/SBThread.i",
                      "/scripts/interface/SBThreadCollection.i",
                      "/scripts/interface/SBType.i",
                      "/scripts/interface/SBTypeCategory.i",
                      "/scripts/interface/SBTypeFilter.i",
                      "/scripts/interface/SBTypeFormat.i",
                      "/scripts/interface/SBTypeNameSpecifier.i",
                      "/scripts/interface/SBTypeSummary.i",
                      "/scripts/interface/SBTypeSynthetic.i",
                      "/scripts/interface/SBValue.i",
                      "/scripts/interface/SBValueList.i",
                      "/scripts/interface/SBWatchpoint.i"]
    bDebug = "-d" in vDictArgs
    strRt = vDictArgs["--srcRoot"]
    strRt = os.path.normcase(strRt)

    strInterfaceFiles = ""
    for strIFace in listIFaceFiles[0: len(listIFaceFiles)]:
        strIFace = os.path.normcase(strIFace)
        strInterfaceFiles += " %s%s" % (strRt, strIFace)

    if bDebug:
        print(strMsgIFaceFiles)
        print(strInterfaceFiles)

    vDictArgs["--ifaceFiles"] = strInterfaceFiles

    return True

#++---------------------------------------------------------------------------
# Details: Compare which file is newer.
# Args:    vFile1 - (R) File name path.
#          vFile2 - (R) File name path.
# Returns: Int - 0 = both not exist, 1 = file 1 newer, 2 = file 2 newer,
#          3 = file 1 not exist.
# Throws:  None.
#--
def which_file_is_newer(vFile1, vFile2):
    bF1 = os.path.exists(vFile1)
    bF2 = os.path.exists(vFile2)
    if bF1 == False and bF2 == False:
        return 0 # Both files not exist
    if bF1 == False:
        return 3 # File 1 not exist
    if bF2 == False:
        return 1 # File 1 is newer / file 2 not exist
    f1Stamp = os.path.getmtime(vFile1)
    f2Stamp = os.path.getmtime(vFile2)
    if f1Stamp > f2Stamp:
        return 1 # File 1 is newer

    return 2 # File 2 is newer than file 1

#++---------------------------------------------------------------------------
# Details: Determine whether the specified file exists.
# Args:    vDictArgs - (R) Program input parameters.
#          vstrFileNamePath - (R) Check this file exists.
# Returns: Bool - True = Files exists, false = not found.
# Throws:  None.
#--
def check_file_exists(vDictArgs, vstrFileNamePath):
    bExists = False
    bDebug = "-d" in vDictArgs

    if os.path.exists(vstrFileNamePath):
        bExists = True
    elif bDebug:
        print((strMsgFileNotExist % vstrFileNamePath))

    return bExists

#++---------------------------------------------------------------------------
# Details: Determine whether the specified file is newer than the
#          LLDBWrapPython.cpp file.
# Args:    vDictArgs - (R) Program input parameters.
#          vstrSwigOpFileNamePath - (R) LLDBWrapPython.cpp file.
#          vstrFileNamePath - (R) Specific file.
# Returns: Bool - True = SWIG update required, false = no update required.
# Throws:  None.
#--
def check_newer_file(vDictArgs, vstrSwigOpFileNamePath, vstrFileNamePath):
    bNeedUpdate = False
    bDebug = "-d" in vDictArgs

    strMsg = ""
    nResult = which_file_is_newer(vstrFileNamePath, vstrSwigOpFileNamePath)
    if nResult == 1:
        strMsg = strMsgFileNewrSwigOpFile % (vstrFileNamePath,
                                             vstrSwigOpFileNamePath)
        bNeedUpdate = True
    elif nResult == 3:
        strMsg = strMsgFileNotExist % vstrFileNamePath
        bNeedUpdate = True

    if bNeedUpdate and bDebug:
        print(strMsg)

    return bNeedUpdate

#++---------------------------------------------------------------------------
# Details: Determine whether the any files in the list are newer than the
#          LLDBWrapPython.cpp file.
# Args:    vDictArgs - (R) Program input parameters.
#          vstrSwigOpFileNamePath - (R) LLDBWrapPython.cpp file.
#          vstrFiles - (R) Multi string file names ' ' delimiter.
# Returns: Bool - True = SWIG update required, false = no update required.
# Throws:  None.
#--
def check_newer_files(vDictArgs, vstrSwigOpFileNamePath, vstrFiles):
    bNeedUpdate = False

    listFiles = vstrFiles.split()
    for strFile in listFiles:
        if check_newer_file(vDictArgs, vstrSwigOpFileNamePath, strFile):
            bNeedUpdate = True
            break

    return bNeedUpdate

#++---------------------------------------------------------------------------
# Details: Retrieve the directory path for Python's dist_packages/
#          site_package folder on a Windows platform.
# Args:    vDictArgs - (R) Program input parameters.
# Returns: Bool - True = function success, False = failure.
#          Str - Python Framework directory path.
#          strErrMsg - Error description on task failure.
# Throws:  None.
#--
def get_framework_python_dir_windows(vDictArgs):
    dbg = utilsDebug.CDebugFnVerbose("Python script get_framework_python_dir_windows()")
    bOk = True
    strWkDir = ""
    strErrMsg = ""

    # We are being built by LLVM, so use the PYTHON_INSTALL_DIR argument,
    # and append the python version directory to the end of it.  Depending
    # on the system other stuff may need to be put here as well.
    from distutils.sysconfig import get_python_lib
    strPythonInstallDir = ""
    bHaveArgPrefix = "--prefix" in vDictArgs
    if bHaveArgPrefix:
        strPythonInstallDir = vDictArgs["--prefix"]
    if strPythonInstallDir.__len__() != 0:
        strWkDir = get_python_lib(True, False, strPythonInstallDir)
    else:
        strWkDir = get_python_lib(True, False)
    strWkDir += "/lldb"
    strWkDir = os.path.normcase(strWkDir)

    return (bOk, strWkDir, strErrMsg)

#++---------------------------------------------------------------------------
# Details: Retrieve the directory path for Python's dist_packages/
#          site_package folder on a UNIX style platform.
# Args:    vDictArgs - (R) Program input parameters.
# Returns: Bool - True = function success, False = failure.
#          Str - Python Framework directory path.
#          strErrMsg - Error description on task failure.
# Throws:  None.
#--
def get_framework_python_dir_other_platforms(vDictArgs):
    dbg = utilsDebug.CDebugFnVerbose("Python script get_framework_python_dir_other_platform()")
    bOk = True
    strWkDir = ""
    strErrMsg = ""
    bDbg = "-d" in vDictArgs

    bMakeFileCalled = "-m" in vDictArgs
    if bMakeFileCalled:
        dbg.dump_text("Built by LLVM")
        return get_framework_python_dir_windows(vDictArgs)
    else:
        dbg.dump_text("Built by XCode")
        # We are being built by XCode, so all the lldb Python files can go
        # into the LLDB.framework/Resources/Python subdirectory.
        strWkDir = vDictArgs["--targetDir"]
        strWkDir += "/LLDB.framework"
        if os.path.exists(strWkDir):
            if bDbg:
                print((strMsgFoundLldbFrameWkDir % strWkDir))
            strWkDir += "/Resources/Python/lldb"
            strWkDir = os.path.normcase(strWkDir)
        else:
            bOk = False
            strErrMsg = strErrMsgFrameWkPyDirNotExist % strWkDir

    return (bOk, strWkDir, strErrMsg)

#++---------------------------------------------------------------------------
# Details: Retrieve the directory path for Python's dist_packages/
#          site_package folder depending on the type of OS platform being
#          used.
# Args:    vDictArgs - (R) Program input parameters.
# Returns: Bool - True = function success, False = failure.
#          Str - Python Framework directory path.
#          strErrMsg - Error description on task failure.
# Throws:  None.
#--
def get_framework_python_dir(vDictArgs):
    dbg = utilsDebug.CDebugFnVerbose("Python script get_framework_python_dir()")
    bOk = True
    strWkDir = ""
    strErrMsg = ""

    eOSType = utilsOsType.determine_os_type()
    if eOSType == utilsOsType.EnumOsType.Unknown:
        bOk = False
        strErrMsg = strErrMsgOsTypeUnknown
    elif eOSType == utilsOsType.EnumOsType.Windows:
        bOk, strWkDir, strErrMsg = get_framework_python_dir_windows(vDictArgs)
    else:
        bOk, strWkDir, strErrMsg = get_framework_python_dir_other_platforms(vDictArgs)

    return (bOk, strWkDir, strErrMsg)

#++---------------------------------------------------------------------------
# Details: Retrieve the configuration build path if present and valid (using
#          parameter --cfgBlddir or copy the Python Framework directory.
# Args:    vDictArgs - (R) Program input parameters.
#          vstrFrameworkPythonDir - (R) Python framework directory.
# Returns: Bool - True = function success, False = failure.
#          Str - Config directory path.
#          strErrMsg - Error description on task failure.
# Throws:  None.
#--
def get_config_build_dir(vDictArgs, vstrFrameworkPythonDir):
    dbg = utilsDebug.CDebugFnVerbose("Python script get_config_build_dir()")
    bOk = True
    strErrMsg = ""

    strConfigBldDir = ""
    bHaveConfigBldDir = "--cfgBldDir" in vDictArgs
    if bHaveConfigBldDir:
        strConfigBldDir = vDictArgs["--cfgBldDir"]
    if (bHaveConfigBldDir == False) or (strConfigBldDir.__len__() == 0):
        strConfigBldDir = vstrFrameworkPythonDir

    return (bOk, strConfigBldDir, strErrMsg)

"""
Removes given file, ignoring error if it doesn't exist.
"""
def remove_ignore_enoent(filename):
    try:
        os.remove(filename)
    except OSError as e:
        import errno
        if e.errno != errno.ENOENT:
            raise

#++---------------------------------------------------------------------------
# Details: Do a SWIG code rebuild. Any number returned by SWIG which is not
#          zero is treated as an error. The generate dependencies flag decides
#          how SWIG is rebuilt and if set false will cause the script to exit
#          immediately with the exit status + 200 if status is not zero.
# Args:    vDictArgs - (R) Program input parameters.
#          vstrSwigDepFile - (R) SWIG dependency file.
#          vstrCfgBldDir - (R) Configuration build directory.
#          vstrSwigOpFile - (R) SWIG output file.
#          vstrSwigIpFile - (R) SWIG input file.
# Returns: Bool - True = function success, False = failure.
#          strMsg - Error or status message.
#          nExitResult - Exit result of SWIG executable.
#                      - 0 = Success.
#                      - 1 = Success, exit this script and parent script.
#                      - +200 = A SWIG error status result.
# Throws:  None.
#--
def do_swig_rebuild(vDictArgs, vstrSwigDepFile, vstrCfgBldDir,
                    vstrSwigOpFile, vstrSwigIpFile):
    dbg = utilsDebug.CDebugFnVerbose("Python script do_swig_rebuild()")
    bOk = True
    strMsg = ""
    bDbg = "-d" in vDictArgs
    bGenDependencies = "-M" in vDictArgs
    strSwig = vDictArgs["--swigExecutable"]
    strSrcRoot = vDictArgs["--srcRoot"]

    strCfg = vstrCfgBldDir
    strOp = vstrSwigOpFile
    strIp = vstrSwigIpFile
    strSi = os.path.normcase("./.")
    strRoot = strSrcRoot + "/include"
    strRoot = os.path.normcase(strRoot)
    strDep = ""
    if bGenDependencies:
        strDep = vstrSwigDepFile + ".tmp"

    # Build the SWIG args list
    strCmd = "%s " % strSwig
    strCmd += "-c++ "
    strCmd += "-shadow "
    strCmd += "-python "
    strCmd += "-threads "
    strCmd += "-I\"%s\" " % strRoot
    strCmd += "-I\"%s\" " % strSi
    strCmd += "-D__STDC_LIMIT_MACROS "
    strCmd += "-D__STDC_CONSTANT_MACROS "
    if bGenDependencies:
        strCmd += "-MMD -MF \"%s\" " % strDep
    strCmd += "-outdir \"%s\" " % strCfg
    strCmd += "-o \"%s\" " % strOp
    strCmd += "\"%s\" " % strIp
    if bDbg:
        print((strMsgSwigExecute % strCmd))

    # Execute SWIG
    process = subprocess.Popen(strCmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=True)
    # Wait for SWIG process to terminate
    strStdOut, strStdErr = process.communicate()
    nResult = process.returncode
    if nResult != 0:
        bOk = False
        nResult += 200
        strMsg = strErrMsgSwigExecute % strStdErr
    else:
        if bDbg and (strStdOut.__len__() != 0):
            strMsg = strStdOut

    if bGenDependencies:
        if bOk:
            if os.path.exists(strDep):
                shutil.move(strDep, vstrSwigDepFile)
        else:
            os.remove(strDep)
            nResult = 1 # Exit this script and parent script
            if bDbg:
                strMsg = strMsgSwigGenDep % strDep
    else:
        strMsg = strMsgSwigNoGenDep + strMsg

    return bOk, strMsg, nResult

#++---------------------------------------------------------------------------
# Details: Execute another Python script from this script in a separate
#          process. No data is passed back to the caller script. It is
#          assumed should any exit result be returned that -ve numbers are
#          error conditions. A zero or +ve numbers mean ok/warning/status.
# Args:    vDictArgs - (R) Program input parameters.
#          vstrArgs - (R) Space separated parameters passed to python.
# Returns: Bool - True = function success, False = failure.
#          strMsg - Error or status message.
# Throws:  None.
#--
def run_python_script(vDictArgs, vstrArgs):
    dbg = utilsDebug.CDebugFnVerbose("Python script run_python_script()")
    bOk = True
    strMsg = ""
    bDbg = "-d" in vDictArgs

    strPy = "%s %s" % (sys.executable, vstrArgs)
    process = subprocess.Popen(strPy, shell=True)
    strStdOut, strStdErr = process.communicate()
    nResult = process.returncode
    if nResult < 0:
        bOk = False
        strErr = strStdErr
        if strErr == None:
            strErr = "No error given"
        strMsg = strErrMsgPythonExecute % (vstrArgs, strErr)
    else:
        if bDbg:
            strOut = strStdOut
            if strOut == None:
                strOut = "No status given"
            strMsg = strOut

    return bOk, strMsg

#++---------------------------------------------------------------------------
# Details: Implement the iterator protocol and/or eq/ne operators for some
#          lldb objects.
#          Append global variable to lldb Python module.
#          And initialize the lldb debugger subsystem.
# Args:    vDictArgs - (R) Program input parameters.
#          vstrCfgBldDir - (R) Configuration build directory.
# Returns: Bool - True = function success, False = failure.
#          strMsg - Error or status message.
# Throws:  None.
#--
def do_modify_python_lldb(vDictArgs, vstrCfgBldDir):
    dbg = utilsDebug.CDebugFnVerbose("Python script do_modify_python_lldb()")
    bOk = True
    strMsg = ""
    bDbg = "-d" in vDictArgs
    strCwd = vDictArgs["--srcRoot"] # /llvm/tools/lldb
    strCwd += "/scripts/Python"
    strPyScript = "modify-python-lldb.py"
    strPath = "%s/%s" % (strCwd, strPyScript)
    strPath = os.path.normcase(strPath)

    bOk = os.path.exists(strPath)
    if not bOk:
        strMsg = strErrMsgModifyPythonLldbPyFileNotFound % (strPyScript, strPath)
        return bOk, strMsg

    strPyArgs = "%s %s" % (strPath, vstrCfgBldDir)
    bOk, strMsg = run_python_script(vDictArgs, strPyArgs)

    return bOk, strMsg

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

""" Details: Program main entry point fn. Called by another Python script.

    --------------------------------------------------------------------------
    Details: This script is to be called by another Python script. It is not
             intended to be called directly i.e from the command line.
             If environmental variable "LLDB_DISABLE_PYTHON" is defined/exists
             it will cause the script to end early creating nothing.
             If environmental variable "GCC_PREPROCESSOR_DEFINITIONS" is
             defined/exists it will cause the script to end early creating
             nothing.
    Args:     vDictArgs - (R) Map of parameter names to values. Used to for the
                              the SWIG required parameters to create code. Note
                              this container does get amended with more data.
            -d (optional)    Determines whether or not this script
                             outputs additional information when running.
            -m (optional)    Specify called from Makefile system. If given locate
                             the LLDBWrapPython.cpp in --srcRoot/source folder
                             else in the --targetDir folder.
            -M (optional)    Specify want SWIG to generate a dependency file.
            --srcRoot        The root of the lldb source tree.
            --targetDir      Where the lldb framework/shared library gets put.
            --cfgBldDir      Where the buildSwigPythonLLDB.py program will
            (optional)       put the lldb.py file it generated from running
                             SWIG.
            --prefix         Is the root directory used to determine where
            (optional)       third-party modules for scripting languages should
                             be installed. Where non-Darwin systems want to put
                             the .py and .so files so that Python can find them
                             automatically. Python install directory.
            --swigExecutable Full path to the SWIG executable. (Determined and
                             passed by buildSwigWrapperClasses.py to here)
    Results:    0 Success
                1 Success, generated dependencies removed
                  LLDBWrapPython.cpp.d.
                -100+ Error from this script to the caller script.
                -100 Error program failure with optional message.
                -200+ - 200 +- the SWIG exit result.

    --------------------------------------------------------------------------

"""
def main(vDictArgs):
    dbg = utilsDebug.CDebugFnVerbose("Python script main()")
    bOk = True
    strMsg = ""

    if not "--swigExecutable" in vDictArgs:
        strErrMsgProgFail += strErrMsgSwigParamsMissing
        return (-100, strErrMsgProgFail)

    bDebug = "-d" in vDictArgs

    strSwigDepFile = ""
    strSwigDepOptions = ""
    bGenDependencies = "-M" in vDictArgs
    if bGenDependencies:
        strSwigDepFile = vDictArgs["--targetDir"] + "/LLDBWrapPython.cpp.d"
        strSwigDepOptions = "-MMD -MF \"%s.tmp\"" % strSwigDepFile
        strSwigDepFile = os.path.normcase(strSwigDepFile)
        strSwigDepOptions = os.path.normcase(strSwigDepOptions)

    bMakeFileCalled = "-m" in vDictArgs
    strSwigOutputFile = ""
    if bMakeFileCalled:
        strSwigOutputFile = vDictArgs["--targetDir"] + "/LLDBWrapPython.cpp"
    else:
        strSwigOutputFile = vDictArgs["--srcRoot"] + "/source/LLDBWrapPython.cpp"
    strSwigOutputFile = os.path.normcase(strSwigOutputFile)

    strRt = vDictArgs["--srcRoot"]
    strSwigInputFile = strRt + "/scripts/lldb.swig"
    strSwigPythonExtensions = strRt + "/scripts/Python/python-extensions.swig"
    strSwigPythonWrapper = strRt + "/scripts/Python/python-wrapper.swig"
    strSwigPythonTypemaps = strRt + "/scripts/Python/python-typemaps.swig"
    strSwigPythonSwigsafecast = strRt + "/scripts/Python/python-swigsafecast.swig"
    strSwigInputFile = os.path.normcase(strSwigInputFile)
    strSwigPythonExtensions = os.path.normcase(strSwigPythonExtensions)
    strSwigPythonWrapper = os.path.normcase(strSwigPythonWrapper)
    strSwigPythonTypemaps = os.path.normcase(strSwigPythonTypemaps)
    strSwigPythonSwigsafecast = os.path.normcase(strSwigPythonSwigsafecast)

    strEnvVarLLDBDisablePython = os.getenv("LLDB_DISABLE_PYTHON", None)
    # We don't want Python for this build, but touch the output file so we
    # don't have to conditionalize the build on this as well.
    # Note, at present iOS doesn't have Python, so if you're building for
    # iOS be sure to set LLDB_DISABLE_PYTHON to 1.
    if (strEnvVarLLDBDisablePython != None) and \
       (strEnvVarLLDBDisablePython == "1"):
        remove_ignore_enoent(strSwigOutputFile)
        open(strSwigOutputFile, 'w').close() # Touch the file
        if bDebug:
            strMsg = strMsgLldbDisablePythonEnv
        return (0, strMsg)

    # If this project is being built with LLDB_DISABLE_PYTHON defined,
    # don't bother generating Python swig bindings -- we don't have
    # Python available.
    strEnvVarGccPreprocessDefs = os.getenv("GCC_PREPROCESSOR_DEFINITIONS",
                                           None)
    if (strEnvVarGccPreprocessDefs != None) or \
       (strEnvVarLLDBDisablePython != None):
        remove_ignore_enoent(strSwigOutputFile)
        open(strSwigOutputFile, 'w').close() # Touch the file
        if bDebug:
            strMsg = strMsgLldbDisableGccEnv
        return (0, strMsg)

    bOk = bOk and get_header_files(vDictArgs)
    bOk = bOk and get_interface_files(vDictArgs)

    strFrameworkPythonDir = ""
    if bOk:
        bNeedUpdate = (check_file_exists(vDictArgs, strSwigOutputFile) == False)
        dbg.dump_object("check_file_exists strSwigOutputFile, bNeedUpdate =", bNeedUpdate)
        if bNeedUpdate == False:
            bNeedUpdate = check_newer_files(vDictArgs, strSwigOutputFile, vDictArgs["--headerFiles"])
            dbg.dump_object("check_newer_files header files than strSwigOutputFile, bNeedUpdate =", bNeedUpdate)
        if bNeedUpdate == False:
            bNeedUpdate = check_newer_files(vDictArgs, strSwigOutputFile, vDictArgs["--ifaceFiles"])
            dbg.dump_object("check_newer_files iface files than strSwigOutputFile, bNeedUpdate =", bNeedUpdate)
        if bNeedUpdate == False:
            bNeedUpdate = check_newer_file(vDictArgs, strSwigOutputFile, strSwigInputFile)
            dbg.dump_object("check_newer_files strSwigInputFile than strSwigOutputFile, bNeedUpdate =", bNeedUpdate)
        if bNeedUpdate == False:
            bNeedUpdate = check_newer_file(vDictArgs, strSwigOutputFile, strSwigPythonExtensions)
            dbg.dump_object("check_newer_files strSwigPythonExtensions than strSwigOutputFile, bNeedUpdate =", bNeedUpdate)
        if bNeedUpdate == False:
            bNeedUpdate = check_newer_file(vDictArgs, strSwigOutputFile, strSwigPythonWrapper)
            dbg.dump_object("check_newer_files strSwigPythonWrapper than strSwigOutputFile, bNeedUpdate =", bNeedUpdate)
        if bNeedUpdate == False:
            bNeedUpdate = check_newer_file(vDictArgs, strSwigOutputFile, strSwigPythonTypemaps)
            dbg.dump_object("check_newer_files strSwigPythonTypemaps than strSwigOutputFile, bNeedUpdate =", bNeedUpdate)
        if bNeedUpdate == False:
            bNeedUpdate = check_newer_file(vDictArgs, strSwigOutputFile, strSwigPythonSwigsafecast)
            dbg.dump_object("check_newer_files strSwigPythonSwigsafecast than strSwigOutputFile, bNeedUpdate =", bNeedUpdate)

        # Determine where to put the files
        bOk, strFrameworkPythonDir, strMsg = get_framework_python_dir(vDictArgs)

    if bOk:
        bOk, strCfgBldDir, strMsg = get_config_build_dir(vDictArgs, strFrameworkPythonDir)

    if bOk and (bNeedUpdate == False):
        strDllPath = strFrameworkPythonDir + "/_lldb.so"
        strDllPath = os.path.normcase(strDllPath)
        bSymbolicLink = check_file_exists(vDictArgs, strDllPath) and os.path.islink(strDllPath)
        bNeedUpdate = not bSymbolicLink
        dbg.dump_object("check_file_exists(vDictArgs, strDllPath) and os.path.islink(strDllPath), bNeedUpdate =", bNeedUpdate)

    if bOk and (bNeedUpdate == False):
        strInitPiPath = strFrameworkPythonDir + "/__init__.py"
        strInitPiPath = os.path.normcase(strInitPiPath)
        print(strInitPiPath)
        bNeedUpdate = not check_file_exists(vDictArgs, strInitPiPath)
        dbg.dump_object("check_file_exists(vDictArgs, strInitPiPath), bNeedUpdate =", bNeedUpdate)

    if bOk:
        if bNeedUpdate == False:
            strMsg = strMsgNotNeedUpdate
            return (0, strMsg)
        else:
            print(strMsgSwigNeedRebuild)
            bOk, strMsg, nExitResult = do_swig_rebuild(vDictArgs, strSwigDepFile,
                                                       strCfgBldDir,
                                                       strSwigOutputFile,
                                                       strSwigInputFile)
            bGenDependencies = "-M" in vDictArgs
            if bGenDependencies == True:
                return (nExitResult, strMsg)

    if bOk:
        bOk, strMsg = do_modify_python_lldb(vDictArgs, strCfgBldDir)

    if bOk:
        return (0, strMsg)
    else:
        strErrMsgProgFail += strMsg
        return (-100, strErrMsgProgFail)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# This script can be called by another Python script by calling the main()
# function directly
if __name__ == "__main__":
    print("Script cannot be called directly, called by buildSwigWrapperClasses.py")
