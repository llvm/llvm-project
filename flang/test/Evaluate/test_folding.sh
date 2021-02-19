#!/usr/bin/env bash
# This script verifies expression folding.
# It compiles a source file with '-fdebug-dump-symbols' and looks for
# parameter declarations to check they have been folded as expected.
# To check folding of an expression EXPR, the fortran program passed to this script
# must contain the following:
#   logical, parameter :: test_x = <compare EXPR to expected value>
# This script will test that all parameter with a name starting with "test_"
# have been folded to .true.
# For instance, acos folding can be tested with:
#
#   real(4), parameter :: res_acos = acos(0.5_4)
#   real(4), parameter :: exp_acos = 1.047
#   logical, parameter :: test_acos = abs(res_acos - exp_acos).LE.(0.001_4)
#
# There are two kinds of failure:
#   - test_x is folded to .false.. This means the expression was folded
#     but the value is not as expected.
#   - test_x is not folded (it is neither .true. nor .false.). This means the
#     compiler could not fold the expression.

if [[ $# < 3 ]]; then
  echo "Usage: $0 <fortran-source> <temp test dir> <f18-executable>"
  exit 1
fi

src=$1
[[ ! -f $src ]] && echo "File not found: $src" && exit 1
shift

temp=$1
mkdir -p $temp
shift

CMD="$* -fdebug-dump-symbols"

# Check if tests should assume folding is using libpgmath
if [[ $LIBPGMATH ]]; then
  CMD="$CMD -DTEST_LIBPGMATH"
  echo "Assuming libpgmath support"
else
  echo "Not assuming libpgmath support"
fi

src1=$temp/symbols.log
src2=$temp/all_parameters.log
src3=$temp/tested_parameters.log
src4=$temp/failures.log
messages=$temp/messages.log
actual_warnings=$temp/actwarnings.log
expected_warnings=$temp/expwarnings.log
warning_diffs=$temp/warnings.diff

if ! ( cd $temp; $CMD $src ) > $src1 2> $messages # compile, dumping symbols
then
  cat $messages
  echo FAIL compilation
  exit 1
fi

# Get all PARAMETER declarations
sed -e '/, PARAMETER/!d' -e 's/, PARAMETER.*init:/ /' \
   -e 's/^ *//'  $src1 > $src2

# Collect test results
sed -e '/^test_/!d' $src2 > $src3

# Check all tests results (keep tests that do not resolve to true)
sed -e '/\.true\._.$/d' $src3 > $src4


#Check warnings
sed -n 's=^[^:]*:\([0-9]*\):[0-9]*: =\1: =p' $messages > $actual_warnings

awk '
  BEGIN { FS = "!WARN: "; }
  /^ *!WARN: / { warnings[nwarnings++] = $2; next; }
  { for (i = 0; i < nwarnings; ++i) printf "%d: %s\n", NR, warnings[i]; nwarnings = 0; }
' $src > $expected_warnings

diff -U0 $actual_warnings $expected_warnings > $warning_diffs

if [ -s $src4 ] || [ -s $warning_diffs ]; then
  echo "folding test failed:"
  # Print failed tests (It will actually print all parameters
  # that have the same suffix as the failed test so that one can get more info
  # by declaring expected_x and result_x for instance)
  if [[ -s $src4 ]]; then
    sed -e 's/test_/_/' -e 's/ .*//' $src4 | grep -f - $src2
  fi
  if [[ -s $warning_diffs ]]; then
    echo "$cmd"
    < $warning_diffs \
      sed -n -e 's/^-\([0-9]\)/actual at \1/p' -e 's/^+\([0-9]\)/expect at \1/p' \
      | sort -n -k 2
  fi
  echo FAIL
  exit 1
else
  passed_results=$(wc -l < $src3)
  passed_warnings=$(wc -l < $expected_warnings)
  passed=$(($passed_warnings + $passed_results))
  echo all $passed tests passed
  echo PASS
fi
