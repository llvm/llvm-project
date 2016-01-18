#!/bin/sh

# build-swig-Python.sh

# SRC_ROOT is the root of the lldb source tree.
# TARGET_DIR is where the lldb framework/shared library gets put.
# CONFIG_BUILD_DIR is where the build-swig-Python-LLDB.sh  shell script
#           put the lldb.py file it was generated from running SWIG.
# PREFIX is the root directory used to determine where third-party modules
#         for scripting languages should be installed.
# debug_flag (optional) determines whether or not this script outputs
#           additional information when running.

SRC_ROOT=$1
TARGET_DIR=$2
CONFIG_BUILD_DIR=$3
PYTHON_INSTALL_DIR=$4
debug_flag=$5
SWIG=$6
makefile_flag=$7
dependency_flag=$8

PYTHON=${PYTHON_EXECUTABLE:-/usr/bin/env python}

if [ -n "$makefile_flag" -a "$makefile_flag" = "-m" ]
then
    MakefileCalled=1
    if [ -n "$dependency_flag" -a "$dependency_flag" = "-M" ]
    then
        GenerateDependencies=1
        swig_depend_file="${TARGET_DIR}/LLDBWrapPython.cpp.d"
        SWIG_DEPEND_OPTIONS="-MMD -MF \"${swig_depend_file}.tmp\""
    else
        GenerateDependencies=0
    fi
else
    MakefileCalled=0
    GenerateDependencies=0
fi

swig_output_file=${TARGET_DIR}/LLDBWrapPython.cpp
swig_input_file=${SRC_ROOT}/scripts/lldb.swig
swig_python_extensions=${SRC_ROOT}/scripts/Python/python-extensions.swig
swig_python_wrapper=${SRC_ROOT}/scripts/Python/python-wrapper.swig
swig_python_typemaps=${SRC_ROOT}/scripts/Python/python-typemaps.swig
swig_python_swigsafecast=${SRC_ROOT}/scripts/Python/python-swigsafecast.swig

if [ "$LLDB_DISABLE_PYTHON" = "1" ] ; then
    # We don't want Python for this build, but touch the output file so we don't have to
    # conditionalize the build on this as well.
    # Note, at present iOS doesn't have Python, so if you're building for iOS be sure to
    # set LLDB_DISABLE_PYTHON to 1.
    rm -rf ${swig_output_file}
    touch ${swig_output_file}

else

if [ -n "$debug_flag" -a "$debug_flag" = "-debug" ]
then
    Debug=1
else
    Debug=0
fi

# If this project is being built with LLDB_DISABLE_PYTHON defined,
# don't bother generating Python swig bindings -- we don't have
# Python available.

if echo $GCC_PREPROCESSOR_DEFINITIONS | grep LLDB_DISABLE_PYTHON
then
  echo "" > "${swig_output_file}"
  exit 0
fi

HEADER_FILES="${SRC_ROOT}/include/lldb/lldb.h"\
" ${SRC_ROOT}/include/lldb/lldb-defines.h"\
" ${SRC_ROOT}/include/lldb/lldb-enumerations.h"\
" ${SRC_ROOT}/include/lldb/lldb-forward.h"\
" ${SRC_ROOT}/include/lldb/lldb-forward-rtti.h"\
" ${SRC_ROOT}/include/lldb/lldb-types.h"\
" ${SRC_ROOT}/include/lldb/API/SBAddress.h"\
" ${SRC_ROOT}/include/lldb/API/SBAttachInfo.h"\
" ${SRC_ROOT}/include/lldb/API/SBBlock.h"\
" ${SRC_ROOT}/include/lldb/API/SBBreakpoint.h"\
" ${SRC_ROOT}/include/lldb/API/SBBreakpointLocation.h"\
" ${SRC_ROOT}/include/lldb/API/SBBroadcaster.h"\
" ${SRC_ROOT}/include/lldb/API/SBCommandInterpreter.h"\
" ${SRC_ROOT}/include/lldb/API/SBCommandReturnObject.h"\
" ${SRC_ROOT}/include/lldb/API/SBCommunication.h"\
" ${SRC_ROOT}/include/lldb/API/SBCompileUnit.h"\
" ${SRC_ROOT}/include/lldb/API/SBData.h"\
" ${SRC_ROOT}/include/lldb/API/SBDebugger.h"\
" ${SRC_ROOT}/include/lldb/API/SBError.h"\
" ${SRC_ROOT}/include/lldb/API/SBEvent.h"\
" ${SRC_ROOT}/include/lldb/API/SBExecutionContext.h"\
" ${SRC_ROOT}/include/lldb/API/SBExpressionOptions.h"\
" ${SRC_ROOT}/include/lldb/API/SBFileSpec.h"\
" ${SRC_ROOT}/include/lldb/API/SBFrame.h"\
" ${SRC_ROOT}/include/lldb/API/SBFunction.h"\
" ${SRC_ROOT}/include/lldb/API/SBHostOS.h"\
" ${SRC_ROOT}/include/lldb/API/SBInstruction.h"\
" ${SRC_ROOT}/include/lldb/API/SBInstructionList.h"\
" ${SRC_ROOT}/include/lldb/API/SBLanguageRuntime.h"\
" ${SRC_ROOT}/include/lldb/API/SBLaunchInfo.h"\
" ${SRC_ROOT}/include/lldb/API/SBLineEntry.h"\
" ${SRC_ROOT}/include/lldb/API/SBListener.h"\
" ${SRC_ROOT}/include/lldb/API/SBModule.h"\
" ${SRC_ROOT}/include/lldb/API/SBModuleSpec.h"\
" ${SRC_ROOT}/include/lldb/API/SBProcess.h"\
" ${SRC_ROOT}/include/lldb/API/SBQueue.h"\
" ${SRC_ROOT}/include/lldb/API/SBQueueItem.h"\
" ${SRC_ROOT}/include/lldb/API/SBSourceManager.h"\
" ${SRC_ROOT}/include/lldb/API/SBStream.h"\
" ${SRC_ROOT}/include/lldb/API/SBStringList.h"\
" ${SRC_ROOT}/include/lldb/API/SBSymbol.h"\
" ${SRC_ROOT}/include/lldb/API/SBSymbolContext.h"\
" ${SRC_ROOT}/include/lldb/API/SBSymbolContextList.h"\
" ${SRC_ROOT}/include/lldb/API/SBTarget.h"\
" ${SRC_ROOT}/include/lldb/API/SBThread.h"\
" ${SRC_ROOT}/include/lldb/API/SBThreadCollection.h"\
" ${SRC_ROOT}/include/lldb/API/SBThreadPlan.h"\
" ${SRC_ROOT}/include/lldb/API/SBType.h"\
" ${SRC_ROOT}/include/lldb/API/SBTypeCategory.h"\
" ${SRC_ROOT}/include/lldb/API/SBTypeFilter.h"\
" ${SRC_ROOT}/include/lldb/API/SBTypeFormat.h"\
" ${SRC_ROOT}/include/lldb/API/SBTypeNameSpecifier.h"\
" ${SRC_ROOT}/include/lldb/API/SBTypeSummary.h"\
" ${SRC_ROOT}/include/lldb/API/SBTypeSynthetic.h"\
" ${SRC_ROOT}/include/lldb/API/SBValue.h"\
" ${SRC_ROOT}/include/lldb/API/SBValueList.h"\
" ${SRC_ROOT}/include/lldb/API/SBVariablesOptions.h"\
" ${SRC_ROOT}/include/lldb/API/SBWatchpoint.h"\
" ${SRC_ROOT}/include/lldb/API/SBUnixSignals.h"

INTERFACE_FILES="${SRC_ROOT}/scripts/interface/SBAddress.i"\
" ${SRC_ROOT}/scripts/interface/SBAttachInfo.i"\
" ${SRC_ROOT}/scripts/interface/SBBlock.i"\
" ${SRC_ROOT}/scripts/interface/SBBreakpoint.i"\
" ${SRC_ROOT}/scripts/interface/SBBreakpointLocation.i"\
" ${SRC_ROOT}/scripts/interface/SBBroadcaster.i"\
" ${SRC_ROOT}/scripts/interface/SBCommandInterpreter.i"\
" ${SRC_ROOT}/scripts/interface/SBCommandReturnObject.i"\
" ${SRC_ROOT}/scripts/interface/SBCommunication.i"\
" ${SRC_ROOT}/scripts/interface/SBCompileUnit.i"\
" ${SRC_ROOT}/scripts/interface/SBData.i"\
" ${SRC_ROOT}/scripts/interface/SBDebugger.i"\
" ${SRC_ROOT}/scripts/interface/SBDeclaration.i"\
" ${SRC_ROOT}/scripts/interface/SBError.i"\
" ${SRC_ROOT}/scripts/interface/SBEvent.i"\
" ${SRC_ROOT}/scripts/interface/SBExecutionContext.i"\
" ${SRC_ROOT}/scripts/interface/SBExpressionOptions.i"\
" ${SRC_ROOT}/scripts/interface/SBFileSpec.i"\
" ${SRC_ROOT}/scripts/interface/SBFrame.i"\
" ${SRC_ROOT}/scripts/interface/SBFunction.i"\
" ${SRC_ROOT}/scripts/interface/SBHostOS.i"\
" ${SRC_ROOT}/scripts/interface/SBInstruction.i"\
" ${SRC_ROOT}/scripts/interface/SBInstructionList.i"\
" ${SRC_ROOT}/scripts/interface/SBLanguageRuntime.i"\
" ${SRC_ROOT}/scripts/interface/SBLaunchInfo.i"\
" ${SRC_ROOT}/scripts/interface/SBLineEntry.i"\
" ${SRC_ROOT}/scripts/interface/SBListener.i"\
" ${SRC_ROOT}/scripts/interface/SBModule.i"\
" ${SRC_ROOT}/scripts/interface/SBModuleSpec.i"\
" ${SRC_ROOT}/scripts/interface/SBPlatform.i"\
" ${SRC_ROOT}/scripts/interface/SBProcess.i"\
" ${SRC_ROOT}/scripts/interface/SBQueue.i"\
" ${SRC_ROOT}/scripts/interface/SBQueueItem.i"\
" ${SRC_ROOT}/scripts/interface/SBSourceManager.i"\
" ${SRC_ROOT}/scripts/interface/SBStream.i"\
" ${SRC_ROOT}/scripts/interface/SBStringList.i"\
" ${SRC_ROOT}/scripts/interface/SBSymbol.i"\
" ${SRC_ROOT}/scripts/interface/SBSymbolContext.i"\
" ${SRC_ROOT}/scripts/interface/SBTarget.i"\
" ${SRC_ROOT}/scripts/interface/SBThread.i"\
" ${SRC_ROOT}/scripts/interface/SBThreadCollection.i"\
" ${SRC_ROOT}/scripts/interface/SBThreadPlan.i"\
" ${SRC_ROOT}/scripts/interface/SBType.i"\
" ${SRC_ROOT}/scripts/interface/SBTypeCategory.i"\
" ${SRC_ROOT}/scripts/interface/SBTypeFilter.i"\
" ${SRC_ROOT}/scripts/interface/SBTypeFormat.i"\
" ${SRC_ROOT}/scripts/interface/SBTypeNameSpecifier.i"\
" ${SRC_ROOT}/scripts/interface/SBTypeSummary.i"\
" ${SRC_ROOT}/scripts/interface/SBTypeSynthetic.i"\
" ${SRC_ROOT}/scripts/interface/SBValue.i"\
" ${SRC_ROOT}/scripts/interface/SBValueList.i"\
" ${SRC_ROOT}/scripts/interface/SBVariablesOptions.i"\
" ${SRC_ROOT}/scripts/interface/SBWatchpoint.i"\
" ${SRC_ROOT}/scripts/interface/SBUnixSignals.i"

if [ $Debug -eq 1 ]
then
    echo "Header files are:"
    echo ${HEADER_FILES}
fi

if [ $Debug -eq 1 ]
then
    echo "SWIG interface files are:"
    echo ${INTERFACE_FILES}
fi

NeedToUpdate=0


if [ ! -f ${swig_output_file} ]
then
    NeedToUpdate=1
    if [ $Debug -eq 1 ]
    then
        echo "Failed to find LLDBWrapPython.cpp"
    fi
fi

if [ $NeedToUpdate -eq 0 ]
then
    for hdrfile in ${HEADER_FILES}
    do
        if [ $hdrfile -nt ${swig_output_file} ]
        then
            NeedToUpdate=1
            if [ $Debug -eq 1 ]
            then
                echo "${hdrfile} is newer than ${swig_output_file}"
                echo "swig file will need to be re-built."
            fi
            break
        fi
    done
fi

if [ $NeedToUpdate -eq 0 ]
then
    for intffile in ${INTERFACE_FILES}
    do
        if [ $intffile -nt ${swig_output_file} ]
        then
            NeedToUpdate=1
            if [ $Debug -eq 1 ]
            then
                echo "${intffile} is newer than ${swig_output_file}"
                echo "swig file will need to be re-built."
            fi
            break
        fi
    done
fi

if [ $NeedToUpdate -eq 0 ]
then
    if [ ${swig_input_file} -nt ${swig_output_file} ]
    then
        NeedToUpdate=1
        if [ $Debug -eq 1 ]
        then
            echo "${swig_input_file} is newer than ${swig_output_file}"
            echo "swig file will need to be re-built."
        fi
    fi
fi

if [ $NeedToUpdate -eq 0 ]
then
    if [ ${swig_python_extensions} -nt ${swig_output_file} ]
    then
        NeedToUpdate=1
        if [ $Debug -eq 1 ]
        then
            echo "${swig_python_extensions} is newer than ${swig_output_file}"
            echo "swig file will need to be re-built."
        fi
    fi
fi

if [ $NeedToUpdate -eq 0 ]
then
    if [ ${swig_python_wrapper} -nt ${swig_output_file} ]
    then
        NeedToUpdate=1
        if [ $Debug -eq 1 ]
        then
            echo "${swig_python_wrapper} is newer than ${swig_output_file}"
            echo "swig file will need to be re-built."
        fi
    fi
fi

if [ $NeedToUpdate -eq 0 ]
then
    if [ ${swig_python_typemaps} -nt ${swig_output_file} ]
    then
        NeedToUpdate=1
        if [ $Debug -eq 1 ]
        then
            echo "${swig_python_typemaps} is newer than ${swig_output_file}"
            echo "swig file will need to be re-built."
        fi
    fi
fi

if [ $NeedToUpdate -eq 0 ]
then
    if [ ${swig_python_swigsafecast} -nt ${swig_output_file} ]
    then
        NeedToUpdate=1
        if [ $Debug -eq 1 ]
        then
            echo "${swig_python_swigsafecast} is newer than ${swig_output_file}"
            echo "swig file will need to be re-built."
        fi
    fi
fi

python_version=`${PYTHON} --version 2>&1 | sed -e 's,Python ,,' -e 's,[.][0-9],,2' -e 's,[a-z][a-z][0-9],,'`

if [ $MakefileCalled -eq 0 ]
then
    framework_python_dir="${TARGET_DIR}/LLDB.framework/Resources/Python/lldb"
else
    if [ -n "${PYTHON_INSTALL_DIR}" ]
    then
        framework_python_dir=`${PYTHON} -c "from distutils.sysconfig import get_python_lib; print get_python_lib(True, False, \"${PYTHON_INSTALL_DIR}\");"`/lldb
    else
        framework_python_dir=`${PYTHON} -c "from distutils.sysconfig import get_python_lib; print get_python_lib(True, False);"`/lldb
    fi
fi

[ -n "${CONFIG_BUILD_DIR}" ] || CONFIG_BUILD_DIR=${framework_python_dir}

if [ ! -L "${framework_python_dir}/_lldb.so" ]
then
    NeedToUpdate=1
fi

if [ ! -f "${framework_python_dir}/__init__.py" ]
then
    NeedToUpdate=1
fi


if [ $NeedToUpdate -eq 0 ]
then
    echo "Everything is up-to-date."
    exit 0
else
    echo "SWIG needs to be re-run."
    if [ -f ${swig_output_file} ]
    then
        rm ${swig_output_file}
    fi
fi


# Build the SWIG C++ wrapper file for Python.

if [ $GenerateDependencies -eq 1 ]
then
    if $SWIG -c++ -shadow -python -threads -I"${SRC_ROOT}/include" -I./. -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -MMD -MF "${swig_depend_file}.tmp" -outdir "${CONFIG_BUILD_DIR}" -o "${swig_output_file}" "${swig_input_file}"
    then
        mv -f "${swig_depend_file}.tmp" "${swig_depend_file}"
    else
        rm -f "${swig_depend_file}.tmp"
        exit 1
    fi
else
    $SWIG -c++ -shadow -python -threads -I"${SRC_ROOT}/include" -I./. -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -outdir "${CONFIG_BUILD_DIR}" -o "${swig_output_file}" "${swig_input_file}" || exit $?
fi

# Implement the iterator protocol and/or eq/ne operators for some lldb objects.
# Append global variable to lldb Python module.
# And initialize the lldb debugger subsystem.
current_dir=`pwd`
if [ -f "${current_dir}/modify-python-lldb.py" ]
then
    ${PYTHON} ${current_dir}/modify-python-lldb.py ${CONFIG_BUILD_DIR}
fi


fi
