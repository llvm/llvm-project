#! /bin/bash

# clang-format-6.0 and above versions are recommended.
# * clang-format < 3.9 cannot be used since SortIncludes is not supported.
#   SortIncludes must be disabled since abti.h depends on the order of #include.

indent_list="clang-format-10 clang-format-9.0 clang-format-8.0 clang-format-7.0 \
             clang-format-6.0 clang-format clang-format-5.0 clang-format-4.0 \
             clang-format-3.9"

is_find_indent=0
for indent in $indent_list; do
    if test ! -z "`which $indent`" ; then
        is_find_indent=1
        break
    fi
done
if [ "is_find_indent" = "x0" ]; then
    echo "clang-format is not found."
    exit 1
fi

indent_code()
{
    file=$1
    if [[ "$file" == *"abt.h.in" ]]; then
        return
    fi
    $indent --style="{BasedOnStyle: llvm, \
                      BreakBeforeBraces: WebKit, \
                      IndentWidth: 4, \
                      Cpp11BracedListStyle: false, \
                      IndentCaseLabels: true, \
                      AlignAfterOpenBracket: Align, \
                      SortIncludes: false, \
                      AllowShortFunctionsOnASingleLine : None, \
                      PenaltyBreakBeforeFirstCallParameter: 100000,
                      SpacesInContainerLiterals: false}" \
                      -i ${file}
}

usage()
{
    echo "Usage: $1 [filename | --all] {--recursive} {--debug}"
    echo "Example 1: format a single file (e.g., src/thread.c)"
    echo "  $1 src/thread.c"
    echo "Example 2: recursively find all files and format them"
    echo "  $1 --all --recursive"
}

# Check usage
if [ -z "$1" ]; then
    usage $0
    exit
fi

# Make sure the parameters make sense
all=0
recursive=0
got_file=0
debug=
ignore=0
ignore_list="__I_WILL_NEVER_FIND_YOU__"
for arg in $@; do
    if [ "$ignore" = "1" ] ; then
        ignore_list="$ignore_list|$arg"
        ignore=0
    continue;
    fi

    if [ "$arg" = "--all" ]; then
        all=1
    elif [ "$arg" = "--recursive" ]; then
        recursive=1
    elif [ "$arg" = "--debug" ]; then
        debug="echo"
    elif [ "$arg" = "--ignore" ] ; then
        ignore=1
    else
        got_file=1
    fi
done

if [ "$recursive" = "1" -a "$all" = "0" ]; then
    echo "--recursive cannot be used without --all"
    usage $0
    exit
fi

if [ "$got_file" = "1" -a "$all" = "1" ]; then
    echo "--all cannot be used in conjunction with a specific file"
    usage $0
    exit
fi

if [ "x$debug" != "x" ]; then
    echo "Use $indent (`$indent --version`)"
fi

if [ "$recursive" = "1" ]; then
    for i in `find . \! -type d | egrep '(\.c$|\.h$|\.c\.in$|\.h\.in$|\.cpp$|\.cpp.in$)' | \
        egrep -v "($ignore_list)"` ; do
        ${debug} indent_code $i
    done
elif [ "$all" = "1" ]; then
    for i in `find . -maxdepth 1 \! -type d | egrep '(\.c$|\.h$|\.c\.in$|\.h\.in$|\.cpp$|\.cpp.in$)' | \
        egrep -v "($ignore_list)"` ; do
        ${debug} indent_code $i
    done
else
    ${debug} indent_code $@
fi
