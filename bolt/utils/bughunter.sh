#!/bin/bash
##===- bolt/utils/bughunter.sh - Help locate BOLT bugs -------*- Script -*-===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# details.
#
##===----------------------------------------------------------------------===##
#
# This script attempts to narrow down llvm-bolt bug to a single function in the
# input binary.
#
# If such a function is found, llvm-bolt could be run just on this function
# to mitigate debugging process.
#
# The following envvars are used by this script:
#
#   BOLT              - path to llvm-bolt
#
#   BOLT_OPTIONS      - options to be used by llvm-bolt
#
#   INPUT_BINARY      - input for llvm-bolt
#
#   PRE_COMMAND       - command to execute prior to running optimized binary
#
#   POST_COMMAND      - command to filter results of running optimized binary
#
#   TIMEOUT_OR_CMD    - optional timeout or command on optimized binary command
#                       if the value is a number with an optional trailing letter
#                       [smhd] it is considered a parameter to "timeout",
#                       otherwise it's a shell command that wraps the optimized
#                       binary command.
#
#   COMMAND_LINE      - command line options to run optimized binary with
#
#   IGNORE_ERROR      - ignore error codes returned from optimized binary
#
#   GOLD_FILE         - file containing expected output from optimized binary
#
#   FUNC_NAMES        - if set, path to an initial list of function names to
#                       search.  Otherwise, nm is used on the original binary.
#
#   OFFLINE           - if set, bughunter will produce the binaries but will not
#                       run them, and will depend on you telling whether it
#                       succeeded or not.
#
#   MAX_FUNCS         - if set, use -max-funcs to narrow down the offending
#                       function.  if non-zero, start -max-funcs at $MAX_FUNCS
#                       otherwise, count the number of symbols in the binary.
#
#   MAX_FUNCS_FLAG    - BOLT command line option to use for MAX_FUNCS search.
#                       Default is -max-funcs.  Can also be used for relocation
#                       debugging, e.g. -max-data-relocations.
#
#   VERBOSE           - if non-empty, set the script to echo mode.
#
##===----------------------------------------------------------------------===##

BOLT=${BOLT:=llvm-bolt}

ulimit -c 0
set -o pipefail

if [[ -n "$VERBOSE" ]]; then
    set -x
fi

if [[ ! -x $INPUT_BINARY ]] ; then
    echo "INPUT_BINARY must be set to an executable file"
    exit 1
fi

if [[ -z "$PRE_COMMAND" ]] ; then
    PRE_COMMAND=':'
fi

if [[ -z "$POST_COMMAND" ]] ; then
    POST_COMMAND='cat'
fi

if [[ -n "$TIMEOUT_OR_CMD" && $TIMEOUT_OR_CMD =~ ^[0-9]+[smhd]?$ ]] ; then
    TIMEOUT_OR_CMD="timeout -s KILL $TIMEOUT_OR_CMD"
fi

if [[ -z "$MAX_FUNCS_FLAG" ]] ; then
    MAX_FUNCS_FLAG="-max-funcs"
fi

OPTIMIZED_BINARY=$(mktemp -t -u --suffix=.bolt $(basename ${INPUT_BINARY}).XXX)
OUTPUT_FILE="${OPTIMIZED_BINARY}.out"
BOLT_LOG=$(mktemp -t -u --suffix=.log boltXXX)

if [[ -z $OFFLINE ]]; then
    echo "Verify input binary passes"
    echo "  INPUT_BINARY: $PRE_COMMAND && $TIMEOUT_OR_CMD $INPUT_BINARY $COMMAND_LINE |& $POST_COMMAND >& $OUTPUT_FILE"
    ($PRE_COMMAND && $TIMEOUT_OR_CMD $INPUT_BINARY $COMMAND_LINE |& $POST_COMMAND >& $OUTPUT_FILE)
    STATUS=$?
    if [[ "$IGNORE_ERROR" == "1" ]]; then
        FAIL=0
    else
        FAIL=$STATUS
    fi
    if [[ -e "$GOLD_FILE" ]] ; then
        cmp -s "$OUTPUT_FILE" "$GOLD_FILE"
        FAIL=$?
    fi
    if [[ $FAIL -ne "0" ]] ; then
        echo "  Warning: input binary failed"
    else
        echo "  Input binary passes."
    fi
fi

echo "Verify optimized binary fails"
($BOLT $BOLT_OPTIONS $INPUT_BINARY -o $OPTIMIZED_BINARY >& $BOLT_LOG)
FAIL=$?
if [[ $FAIL -eq "0" ]]; then
    if [[ -z $OFFLINE ]]; then
        echo "  OPTIMIZED_BINARY: $PRE_COMMAND && $TIMEOUT_OR_CMD $OPTIMIZED_BINARY $COMMAND_LINE |& $POST_COMMAND >& $OUTPUT_FILE"
        ($PRE_COMMAND && $TIMEOUT_OR_CMD $OPTIMIZED_BINARY $COMMAND_LINE |& $POST_COMMAND >& $OUTPUT_FILE)
        STATUS=$?
        if [[ "$IGNORE_ERROR" == "1" ]]; then
            FAIL=0
        else
            FAIL=$STATUS
        fi
        if [[ -e "$GOLD_FILE" ]] ; then
            cmp -s "$OUTPUT_FILE" "$GOLD_FILE"
            FAIL=$?
        fi
    else
        echo "Did it pass? Type the return code [0 = pass, 1 = fail]"
        read -n1 FAIL
    fi
    if [[ $FAIL -eq "0" ]] ; then
        echo "  Warning: optimized binary passes."
    else
        echo "  Optimized binary fails as expected."
    fi
else
    echo "  Bolt crashes while generating optimized binary."
fi

# Collect function names
FF=$(mktemp -t -u --suffix=.txt func-names.XXX)
nm --defined-only -p $INPUT_BINARY | grep " [TtWw] " | cut -d ' ' -f 3 | egrep -v "\._" | egrep -v '^$' | sort -u > $FF

# Use function names or numbers
if [[ -z "$MAX_FUNCS" ]] ; then
    # Do binary search on function names
    if [[ -n "$FUNC_NAMES" ]]; then
        FF=$FUNC_NAMES
    fi
    NUM_FUNCS=$(wc -l $FF | cut -d ' ' -f 1)
    HALF=$(expr \( $NUM_FUNCS + 1 \) / 2)
    PREFIX=$(mktemp -t -u --suffix=.txt func-names.XXX)
    FF0=$PREFIX\aa
    FF1=$PREFIX\ab
    split -a 2 -l $HALF $FF $PREFIX
    FF=$FF0
    NUM_FUNCS=$(wc -l $FF | cut -d ' ' -f 1)
    CONTINUE=$(expr $NUM_FUNCS \> 0)
else
    P=0
    if [[ "$MAX_FUNCS" -eq "0" ]]; then
        Q=$(wc -l $FF | cut -d ' ' -f 1)
    else
        Q=$MAX_FUNCS
    fi
    I=$Q
    CONTINUE=$(expr \( $Q - $P \) \> 1)
fi

ITER=0
while [[ "$CONTINUE" -ne "0" ]] ; do
    rm -f $OPTIMIZED_BINARY
    if [[ -z "$MAX_FUNCS" ]] ; then
        echo "Iteration $ITER, trying $FF / $HALF functions"
        SEARCH_OPT="-funcs-file-no-regex=$FF"
    else
        I=$(expr \( $Q + $P \) / 2)
        echo "Iteration $ITER, P=$P, Q=$Q, I=$I"
        SEARCH_OPT="$MAX_FUNCS_FLAG=$I"
    fi
    echo "  BOLT: $BOLT $BOLT_OPTIONS $INPUT_BINARY $SEARCH_OPT -o $OPTIMIZED_BINARY >& $BOLT_LOG"
    ($BOLT $BOLT_OPTIONS $INPUT_BINARY $SEARCH_OPT -o $OPTIMIZED_BINARY >& $BOLT_LOG)
    FAIL=$?
    echo "  BOLT failure=$FAIL"
    rm -f $OUTPUT_FILE
    if [[ $FAIL -eq "0" ]] ; then
        if [[ -z $OFFLINE ]]; then
            echo "  OPTIMIZED_BINARY: $PRE_COMMAND && $TIMEOUT_OR_CMD $OPTIMIZED_BINARY $COMMAND_LINE |& $POST_COMMAND >& $OUTPUT_FILE"
            ($PRE_COMMAND && $TIMEOUT_OR_CMD $OPTIMIZED_BINARY $COMMAND_LINE |& $POST_COMMAND >& $OUTPUT_FILE)
            STATUS=$?
            if [[ "$IGNORE_ERROR" == "1" ]]; then
                FAIL=0
            else
                FAIL=$STATUS
            fi
            if [[ -e "$GOLD_FILE" ]] ; then
                cmp -s "$OUTPUT_FILE" "$GOLD_FILE"
                FAIL=$?
            fi
            echo "  OPTIMIZED_BINARY failure=$FAIL"
        else
            echo "Did it pass? Type the return code [0 = pass, 1 = fail]"
            read -n1 FAIL
        fi
    else
        FAIL=1
    fi

    if [[ -z "$MAX_FUNCS" ]] ; then
        if [[ $FAIL -eq "0" ]] ; then
            if [[ "$FF" == "$FF1" ]]; then
                NUM_FUNCS=0
                break;
            fi
            FF=$FF1
            NUM_FUNCS=$(wc -l $FF | cut -d ' ' -f 1)
        else
            HALF=$(expr \( $NUM_FUNCS + 1 \) / 2)
            PREFIX=$(mktemp -t -u --suffix=.txt func-names.XXX)
            split -a 2 -l $HALF $FF $PREFIX
            FF0=$PREFIX\aa
            FF1=$PREFIX\ab
            FF=$FF0
            NUM_FUNCS=$(wc -l $FF | cut -d ' ' -f 1)
            if [[ $NUM_FUNCS -eq "1" && ! -e $FF1 ]]; then
                break;
            fi
        fi
        CONTINUE=$(expr $NUM_FUNCS \> 0)
    else
        if [[ $FAIL -eq "0" ]] ; then
            P=$I
        else
            Q=$I
        fi
        FF=$I
        HALF=$I
        CONTINUE=$(expr \( $Q - $P \) \> 1)
    fi
    ITER=$(expr $ITER + 1)
done

if [[ -z "$MAX_FUNCS" ]] ; then
    if [[ "$NUM_FUNCS" -ne "0" ]] ; then
        FAILED="The function(s) that failed are in $FF"
    fi
else
    if [[ $P -ne $Q ]] ; then
        FF=$(grep "processing ending" $BOLT_LOG | sed -e "s/BOLT-INFO: processing ending on \(.*\)/\1/g" | tail -1)
        FAILED="The item that failed is $FF @ $Q"
    fi
fi

if [[ -n "$FAILED" ]] ; then
    echo "$FAILED"
    echo "To reproduce, run: $BOLT $BOLT_OPTIONS $INPUT_BINARY $SEARCH_OPT -o $OPTIMIZED_BINARY"
else
    echo "Unable to reproduce bug."
fi

rm $OPTIMIZED_BINARY $OUTPUT_FILE $BOLT_LOG
