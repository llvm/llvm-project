#! /bin/sh

# finish-swig-wrapper-classes.sh
#
# For each scripting language liblldb supports, we need to create the
# appropriate Script Bridge wrapper classes for that language so that
# users can call Script Bridge functions from within the script interpreter.
#
# We use SWIG to create a C++ file containing the appropriate wrapper classes
# and funcitons for each scripting language, before liblldb is built (thus
# the C++ file can be compiled into liblldb.  In some cases, additional work
# may need to be done after liblldb has been compiled, to make the scripting
# language stuff fully functional.  Any such post-processing is handled through
# the shell scripts called here.

# SRC_ROOT is the root of the lldb source tree.
# TARGET_DIR is where the lldb framework/shared library gets put.
# CONFIG_BUILD_DIR is where the build-swig-Python-LLDB.sh  shell script
#           put the lldb.py file it generated from running SWIG.
# PREFIX is the root directory used to determine where third-party modules
#         for scripting languages should be installed.
# debug_flag (optional) determines whether or not this script outputs
#           additional information when running.

SRC_ROOT=$1
TARGET_DIR=$2
CONFIG_BUILD_DIR=$3
PREFIX=$4

shift 4

if [ -n "$1" -a "$1" = "-debug" ]
then
    debug_flag=$1
    Debug=1
    shift
else
    debug_flag=""
    Debug=0
fi

if [ -n "$1" -a "$1" = "-m" ]
then
    makefile_flag="$1"
    shift
else
    makefile_flag=""
fi

#
# For each scripting language, see if a post-processing script for that
# language exists, and if so, call it.
#
# For now the only language we support is Python, but we expect this to
# change.

languages="Python"
cwd=${SRC_ROOT}/scripts

for curlang in $languages
do
    if [ $Debug -eq 1 ]
    then
        echo "Current language is $curlang"
    fi

    if [ ! -d "$cwd/$curlang" ]
    then
        echo "error:  unable to find $curlang script sub-dirctory" >&2
        continue
    else

        if [ $Debug -eq 1 ]
        then
            echo "Found $curlang sub-directory"
        fi

        cd $cwd/$curlang

        filename="./finish-swig-${curlang}-LLDB.sh"

        if [ -f $filename ]
        then
            if [ $Debug -eq 1 ]
            then
                echo "Found $curlang post-processing script for LLDB"
                echo "Executing $curlang post-processing script..."
            fi


            ./finish-swig-${curlang}-LLDB.sh $SRC_ROOT $TARGET_DIR $CONFIG_BUILD_DIR "${PREFIX}" "${debug_flag}" "${makefile_flag}"
        fi
    fi
done

exit 0
