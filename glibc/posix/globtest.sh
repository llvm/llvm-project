#!/bin/bash
# Test for glob(3).
# Copyright (C) 1997-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

set -e

common_objpfx=$1; shift
test_via_rtld_prefix=$1; shift
test_program_prefix=$1; shift
test_wrapper_env=$1; shift
logfile=$common_objpfx/posix/globtest.out

#CMP=cmp
CMP="diff -u"

# We have to make the paths `common_objpfx' absolute.
case "$common_objpfx" in
  .*)
    common_objpfx="`pwd`/$common_objpfx"
    ;;
  *)
    ;;
esac

# Since we use `sort' we must make sure to use the same locale everywhere.
LC_ALL=C
export LC_ALL

# Create the arena
testdir=${common_objpfx}posix/globtest-dir
testout=${common_objpfx}posix/globtest-out
rm -rf $testdir $testout
mkdir $testdir

cleanup() {
    chmod 777 $testdir/noread
    rm -fr $testdir $testout
}

trap cleanup 0 HUP INT QUIT TERM

echo 1 > $testdir/file1
echo 2 > $testdir/file2
echo 3 > $testdir/-file3
echo 4 > $testdir/~file4
echo 5 > $testdir/.file5
echo 6 > $testdir/'*file6'
echo 7 > $testdir/'{file7,}'
echo 8 > $testdir/'\{file8\}'
echo 9 > $testdir/'\{file9\,file9b\}'
echo 9 > $testdir/'\file9b\' #'
echo a > $testdir/'filea,'
echo a > $testdir/'fileb}c'
mkdir $testdir/dir1
mkdir $testdir/dir2
test -d $testdir/noread || mkdir $testdir/noread
chmod a-r $testdir/noread
echo 1_1 > $testdir/dir1/file1_1
echo 1_2 > $testdir/dir1/file1_2
ln -fs dir1 $testdir/link1

# Run some tests.
result=0
rm -f $logfile

# Normal test
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`*file6'
`-file3'
`\file9b\'
`\{file8\}'
`\{file9\,file9b\}'
`dir1'
`dir2'
`file1'
`file2'
`filea,'
`fileb}c'
`link1'
`noread'
`{file7,}'
`~file4'
EOF
if test $failed -ne 0; then
  echo "Normal test failed" >> $logfile
  result=1
fi

# Don't let glob sort it
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -s "$testdir" "*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`*file6'
`-file3'
`\file9b\'
`\{file8\}'
`\{file9\,file9b\}'
`dir1'
`dir2'
`file1'
`file2'
`filea,'
`fileb}c'
`link1'
`noread'
`{file7,}'
`~file4'
EOF
if test $failed -ne 0; then
  echo "No sort test failed" >> $logfile
  result=1
fi

# Mark directories
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -m "$testdir" "*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`*file6'
`-file3'
`\file9b\'
`\{file8\}'
`\{file9\,file9b\}'
`dir1/'
`dir2/'
`file1'
`file2'
`filea,'
`fileb}c'
`link1/'
`noread/'
`{file7,}'
`~file4'
EOF
if test $failed -ne 0; then
  echo "Mark directories test failed" >> $logfile
  result=1
fi

# Find files starting with .
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -p "$testdir" "*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`*file6'
`-file3'
`.'
`..'
`.file5'
`\file9b\'
`\{file8\}'
`\{file9\,file9b\}'
`dir1'
`dir2'
`file1'
`file2'
`filea,'
`fileb}c'
`link1'
`noread'
`{file7,}'
`~file4'
EOF
if test $failed -ne 0; then
  echo "Leading period test failed" >> $logfile
  result=1
fi

# Test braces
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -b "$testdir" "file{1,2}" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`file1'
`file2'
EOF
if test $failed -ne 0; then
  echo "Braces test failed" >> $logfile
  result=1
fi

failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -b "$testdir" "{file{1,2},-file3}" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`-file3'
`file1'
`file2'
EOF
if test $failed -ne 0; then
  echo "Braces test 2 failed" >> $logfile
  result=1
fi

failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -b "$testdir" "{" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "Braces test 3 failed" >> $logfile
  result=1
fi

# Test NOCHECK
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -c "$testdir" "abc" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`abc'
EOF
if test $failed -ne 0; then
  echo "No check test failed" >> $logfile
  result=1
fi

# Test NOMAGIC without magic characters
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -g "$testdir" "abc" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`abc'
EOF
if test $failed -ne 0; then
  echo "No magic test failed" >> $logfile
  result=1
fi

# Test NOMAGIC with magic characters
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -g "$testdir" "abc*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "No magic w/ magic chars test failed" >> $logfile
  result=1
fi

# Test NOMAGIC for subdirs
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -g "$testdir" "*/does-not-exist" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "No magic in subdir test failed" >> $logfile
  result=1
fi

# Test subdirs correctly
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "*/*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`dir1/file1_1'
`dir1/file1_2'
`link1/file1_1'
`link1/file1_2'
EOF
if test $failed -ne 0; then
  echo "Subdirs test failed" >> $logfile
  result=1
fi

# Test subdirs for invalid names
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "*/1" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "Invalid subdir test failed" >> $logfile
  result=1
fi

# Test subdirs with wildcard
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "*/*1_1" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`dir1/file1_1'
`link1/file1_1'
EOF
if test $failed -ne 0; then
  echo "Wildcard subdir test failed" >> $logfile
  result=1
fi

# Test subdirs with ?
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "*/*?_?" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`dir1/file1_1'
`dir1/file1_2'
`link1/file1_1'
`link1/file1_2'
EOF
if test $failed -ne 0; then
  echo "Wildcard2 subdir test failed" >> $logfile
  result=1
fi

failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "*/file1_1" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`dir1/file1_1'
`link1/file1_1'
EOF
if test $failed -ne 0; then
  echo "Wildcard3 subdir test failed" >> $logfile
  result=1
fi

failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "*-/*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "Wildcard4 subdir test failed" >> $logfile
  result=1
fi

failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "*-" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "Wildcard5 subdir test failed" >> $logfile
  result=1
fi

# Test subdirs with ?
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "*/*?_?" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`dir1/file1_1'
`dir1/file1_2'
`link1/file1_1'
`link1/file1_2'
EOF
if test $failed -ne 0; then
  echo "Wildcard6 subdir test failed" >> $logfile
  result=1
fi

# Test subdirs with [ .. ]
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "*/file1_[12]" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`dir1/file1_1'
`dir1/file1_2'
`link1/file1_1'
`link1/file1_2'
EOF
if test $failed -ne 0; then
  echo "Brackets test failed" >> $logfile
  result=1
fi

# Test ']' inside bracket expression
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "dir1/file1_[]12]" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`dir1/file1_1'
`dir1/file1_2'
EOF
if test $failed -ne 0; then
  echo "Brackets2 test failed" >> $logfile
  result=1
fi

# Test tilde expansion
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -q -t "$testdir" "~" |
sort >$testout
echo ~ | $CMP - $testout >> $logfile || failed=1
if test $failed -ne 0; then
  if test -d ~; then
    echo "Tilde test failed" >> $logfile
    result=1
  else
    echo "Tilde test could not be run" >> $logfile
  fi
fi

# Test tilde expansion with trailing slash
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -q -t "$testdir" "~/" |
sort > $testout
# Some shell incorrectly(?) convert ~/ into // if ~ expands to /.
if test ~/ = //; then
    echo / | $CMP - $testout >> $logfile || failed=1
else
    echo ~/ | $CMP - $testout >> $logfile || failed=1
fi
if test $failed -ne 0; then
  if test -d ~/; then
    echo "Tilde2 test failed" >> $logfile
    result=1
  else
    echo "Tilde2 test could not be run" >> $logfile
  fi
fi

# Test tilde expansion with username
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -q -t "$testdir" "~"$USER |
sort > $testout
eval echo ~$USER | $CMP - $testout >> $logfile || failed=1
if test $failed -ne 0; then
  if eval test -d ~$USER; then
    echo "Tilde3 test failed" >> $logfile
    result=1
  else
    echo "Tilde3 test could not be run" >> $logfile
  fi
fi

# Tilde expansion shouldn't match a file
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -T "$testdir" "~file4" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "Tilde4 test failed" >> $logfile
  result=1
fi

# Matching \** should only find *file6
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "\**" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`*file6'
EOF
if test $failed -ne 0; then
  echo "Star test failed" >> $logfile
  result=1
fi

# ... unless NOESCAPE is used, in which case it should entries with a
# leading \.
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -e "$testdir" "\**" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`\file9b\'
`\{file8\}'
`\{file9\,file9b\}'
EOF
if test $failed -ne 0; then
  echo "Star2 test failed" >> $logfile
  result=1
fi

# Matching \*file6 should find *file6
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "\*file6" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`*file6'
EOF
if test $failed -ne 0; then
  echo "Star3 test failed" >> $logfile
  result=1
fi

# GLOB_BRACE alone
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -b "$testdir" '\{file7\,\}' |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`{file7,}'
EOF
if test $failed -ne 0; then
  echo "Brace4 test failed" >> $logfile
  result=1
fi

# GLOB_BRACE and GLOB_NOESCAPE
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -b -e "$testdir" '\{file9\,file9b\}' |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`\file9b\'
EOF
if test $failed -ne 0; then
  echo "Brace5 test failed" >> $logfile
  result=1
fi

# Escaped comma
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -b "$testdir" '{filea\,}' |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`filea,'
EOF
if test $failed -ne 0; then
  echo "Brace6 test failed" >> $logfile
  result=1
fi

# Escaped closing brace
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -b "$testdir" '{fileb\}c}' |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`fileb}c'
EOF
if test $failed -ne 0; then
  echo "Brace7 test failed" >> $logfile
  result=1
fi

# Try a recursive failed search
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -e "$testdir" "a*/*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "Star4 test failed" >> $logfile
  result=1
fi

# ... with GLOB_ERR
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -E "$testdir" "a*/*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "Star5 test failed" >> $logfile
  result=1
fi

# Try a recursive search in unreadable directory
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "noread/*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "Star6 test failed" >> $logfile
  result=1
fi

failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "noread*/*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_NOMATCH
EOF
if test $failed -ne 0; then
  echo "Star6 test failed" >> $logfile
  result=1
fi

# The following tests will fail if run as root.
user=`id -un 2> /dev/null`
if test -z "$user"; then
    uid="$USER"
fi
if test "$user" != root; then
    # ... with GLOB_ERR
    ${test_program_prefix} \
    ${common_objpfx}posix/globtest -E "$testdir" "noread/*" |
    sort > $testout
    cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_ABORTED
EOF

    ${test_program_prefix} \
    ${common_objpfx}posix/globtest -E "$testdir" "noread*/*" |
    sort > $testout
    cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
GLOB_ABORTED
EOF
if test $failed -ne 0; then
  echo "GLOB_ERR test failed" >> $logfile
  result=1
fi
fi # not run as root

# Try multiple patterns (GLOB_APPEND)
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest "$testdir" "file1" "*/*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`dir1/file1_1'
`dir1/file1_2'
`file1'
`link1/file1_1'
`link1/file1_2'
EOF
if test $failed -ne 0; then
  echo "GLOB_APPEND test failed" >> $logfile
  result=1
fi

# Try multiple patterns (GLOB_APPEND) with offset (GLOB_DOOFFS)
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -o "$testdir" "file1" "*/*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`abc'
`dir1/file1_1'
`dir1/file1_2'
`file1'
`link1/file1_1'
`link1/file1_2'
EOF
if test $failed -ne 0; then
  echo "GLOB_APPEND2 test failed" >> $logfile
  result=1
fi

# Test NOCHECK with non-existing file in subdir.
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -c "$testdir" "*/blahblah" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`*/blahblah'
EOF
if test $failed -ne 0; then
  echo "No check2 test failed" >> $logfile
  result=1
fi

# Test [[:punct:]] not matching leading period.
failed=0
${test_program_prefix} \
${common_objpfx}posix/globtest -c "$testdir" "[[:punct:]]*" |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`*file6'
`-file3'
`\file9b\'
`\{file8\}'
`\{file9\,file9b\}'
`{file7,}'
`~file4'
EOF
if test $failed -ne 0; then
  echo "Punct test failed" >> $logfile
  result=1
fi

mkdir $testdir/'dir3*'
echo 1 > $testdir/'dir3*'/file1
mkdir $testdir/'dir4[a'
echo 2 > $testdir/'dir4[a'/file1
echo 3 > $testdir/'dir4[a'/file2
mkdir $testdir/'dir5[ab]'
echo 4 > $testdir/'dir5[ab]'/file1
echo 5 > $testdir/'dir5[ab]'/file2
mkdir $testdir/dir6
echo 6 > $testdir/dir6/'file1[a'
echo 7 > $testdir/dir6/'file1[ab]'
failed=0
v=`${test_program_prefix} \
   ${common_objpfx}posix/globtest "$testdir" 'dir3\*/file2'`
test "$v" != 'GLOB_NOMATCH' && echo "$v" >> $logfile && failed=1
${test_program_prefix} \
${common_objpfx}posix/globtest -c "$testdir" \
'dir3\*/file1' 'dir3\*/file2' 'dir1/file\1_1' 'dir1/file\1_9' \
'dir2\/' 'nondir\/' 'dir4[a/fil*1' 'di*r4[a/file2' 'dir5[ab]/file[12]' \
'dir6/fil*[a' 'dir*6/file1[a' 'dir6/fi*l[ab]' 'dir*6/file1[ab]' \
'dir6/file1[[.a.]*' |
sort > $testout
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`dir*6/file1[ab]'
`dir1/file1_1'
`dir1/file\1_9'
`dir2/'
`dir3*/file1'
`dir3\*/file2'
`dir4[a/file1'
`dir4[a/file2'
`dir5[ab]/file[12]'
`dir6/fi*l[ab]'
`dir6/file1[a'
`dir6/file1[a'
`dir6/file1[a'
`dir6/file1[ab]'
`nondir\/'
EOF
${test_wrapper_env} \
HOME="$testdir" \
${test_via_rtld_prefix} \
${common_objpfx}posix/globtest -ct "$testdir" \
'~/dir1/file1_1' '~/dir1/file1_9' '~/dir3\*/file1' '~/dir3\*/file2' \
'~\/dir1/file1_2' |
sort > $testout
cat <<EOF | $CMP - $testout >> $logfile || failed=1
\`$testdir/dir1/file1_1'
\`$testdir/dir1/file1_2'
\`$testdir/dir3*/file1'
\`~/dir1/file1_9'
\`~/dir3\\*/file2'
EOF
if eval test -d ~"$USER"/; then
  user=`echo "$USER" | sed -n -e 's/^\([^\\]\)\([^\\][^\\]*\)$/~\1\\\\\2/p'`
  if test -n "$user"; then
    ${test_program_prefix} \
    ${common_objpfx}posix/globtest -ctq "$testdir" "$user/" |
    sort > $testout
    eval echo ~$USER/ | $CMP - $testout >> $logfile || failed=1
    ${test_program_prefix} \
    ${common_objpfx}posix/globtest -ctq "$testdir" "$user\\/" |
    sort > $testout
    eval echo ~$USER/ | $CMP - $testout >> $logfile || failed=1
    ${test_program_prefix} \
    ${common_objpfx}posix/globtest -ctq "$testdir" "$user" |
    sort > $testout
    eval echo ~$USER | $CMP - $testout >> $logfile || failed=1
  fi
fi
if test $failed -ne 0; then
  echo "Escape tests failed" >> $logfile
  result=1
fi

# Test GLOB_BRACE and GLIB_DOOFFS with malloc checking
failed=0
${test_wrapper_env} \
MALLOC_PERTURB_=65 \
${test_via_rtld_prefix} \
${common_objpfx}posix/globtest -b -o "$testdir" "file{1,2}" > $testout || failed=1
cat <<"EOF" | $CMP - $testout >> $logfile || failed=1
`abc'
`file1'
`file2'
EOF
if test $failed -ne 0; then
  echo "GLOB_BRACE+GLOB_DOOFFS test failed" >> $logfile
  result=1
fi

if test $result -eq 0; then
    echo "All OK." > $logfile
fi

exit $result

# Preserve executable bits for this shell script.
Local Variables:
eval:(defun frobme () (set-file-modes buffer-file-name file-mode))
eval:(make-local-variable 'file-mode)
eval:(setq file-mode (file-modes (buffer-file-name)))
eval:(make-local-variable 'after-save-hook)
eval:(add-hook 'after-save-hook 'frobme)
End:
