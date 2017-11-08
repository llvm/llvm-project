#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Guess path to LLVM_CHECKOUT if not provided
if [ "${LLVM_CHECKOUT}" = "" ]; then
  LLVM_CHECKOUT="${SCRIPT_DIR}/../../../../../"
fi

# python tools setup
CPPLINT=${SCRIPT_DIR}/cpplint.py
LITLINT=${SCRIPT_DIR}/litlint.py
if [ "${PYTHON_EXECUTABLE}" != "" ]; then
  CPPLINT="${PYTHON_EXECUTABLE} ${CPPLINT}"
  LITLINT="${PYTHON_EXECUTABLE} ${LITLINT}"
fi

# Filters
# TODO: remove some of these filters
COMMON_LINT_FILTER=-build/include,-build/header_guard,-legal/copyright,-whitespace/comments,-readability/casting,\
-build/namespaces,-readability/braces
ASAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int
ASAN_TEST_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/sizeof,-runtime/int,-runtime/printf,-runtime/threadsafe_fn
ASAN_LIT_TEST_LINT_FILTER=${ASAN_TEST_LINT_FILTER},-whitespace/line_length
TSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
TSAN_TEST_LINT_FILTER=${TSAN_RTL_LINT_FILTER},-runtime/threadsafe_fn,-runtime/int
TSAN_LIT_TEST_LINT_FILTER=${TSAN_TEST_LINT_FILTER},-whitespace/line_length
MSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
LSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
LSAN_LIT_TEST_LINT_FILTER=${LSAN_RTL_LINT_FILTER},-whitespace/line_length
DFSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int,-runtime/printf,-runtime/references,-readability/function
SCUDO_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
COMMON_RTL_INC_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int,-runtime/sizeof,-runtime/printf,-readability/fn_size
SANITIZER_INCLUDES_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int

MKTEMP_DIR=$(mktemp -qd /tmp/check_lint.XXXXXXXXXX)
MKTEMP="mktemp -q ${MKTEMP_DIR}/tmp.XXXXXXXXXX"
cleanup() {
  rm -rf $MKTEMP_DIR
}
trap cleanup EXIT

cd ${LLVM_CHECKOUT}

EXITSTATUS=0
ERROR_LOG=$(${MKTEMP})

run_lint() {
  FILTER=$1
  shift
  TASK_LOG=$(${MKTEMP})
  ${CPPLINT} --filter=${FILTER} "$@" 2>$TASK_LOG
  if [ "$?" != "0" ]; then
    cat $TASK_LOG | grep -v "Done processing" | grep -v "Total errors found" \
      | grep -v "Skipping input" >> $ERROR_LOG
  fi
  if [ "${SILENT}" != "1" ]; then
    cat $TASK_LOG
  fi
  ${LITLINT} "$@" 2>>$ERROR_LOG
}

if [ "${COMPILER_RT}" = "" ]; then
  COMPILER_RT=projects/compiler-rt
fi
LIT_TESTS=${COMPILER_RT}/test
# Headers
SANITIZER_INCLUDES=${COMPILER_RT}/include/sanitizer
run_lint ${SANITIZER_INCLUDES_LINT_FILTER} ${SANITIZER_INCLUDES}/*.h &

# Sanitizer_common
COMMON_RTL=${COMPILER_RT}/lib/sanitizer_common
run_lint ${COMMON_RTL_INC_LINT_FILTER} ${COMMON_RTL}/*.cc \
                                       ${COMMON_RTL}/*.h \
                                       ${COMMON_RTL}/tests/*.cc &

# Interception
INTERCEPTION=${COMPILER_RT}/lib/interception
run_lint ${ASAN_RTL_LINT_FILTER} ${INTERCEPTION}/*.cc \
                                 ${INTERCEPTION}/*.h &

# ASan
ASAN_RTL=${COMPILER_RT}/lib/asan
run_lint ${ASAN_RTL_LINT_FILTER} ${ASAN_RTL}/*.cc \
                                 ${ASAN_RTL}/*.h &
run_lint ${ASAN_TEST_LINT_FILTER} ${ASAN_RTL}/tests/*.cc \
                                  ${ASAN_RTL}/tests/*.h &
run_lint ${ASAN_LIT_TEST_LINT_FILTER} ${LIT_TESTS}/asan/*/*.cc &

# TSan
TSAN_RTL=${COMPILER_RT}/lib/tsan
run_lint ${TSAN_RTL_LINT_FILTER} ${TSAN_RTL}/rtl/*.cc \
                                 ${TSAN_RTL}/rtl/*.h &
run_lint ${TSAN_TEST_LINT_FILTER} ${TSAN_RTL}/tests/rtl/*.cc \
                                  ${TSAN_RTL}/tests/rtl/*.h \
                                  ${TSAN_RTL}/tests/unit/*.cc &
run_lint ${TSAN_LIT_TEST_LINT_FILTER} ${LIT_TESTS}/tsan/*.cc &

# MSan
MSAN_RTL=${COMPILER_RT}/lib/msan
run_lint ${MSAN_RTL_LINT_FILTER} ${MSAN_RTL}/*.cc \
                                 ${MSAN_RTL}/*.h &

# LSan
LSAN_RTL=${COMPILER_RT}/lib/lsan
run_lint ${LSAN_RTL_LINT_FILTER} ${LSAN_RTL}/*.cc \
                                 ${LSAN_RTL}/*.h &
run_lint ${LSAN_LIT_TEST_LINT_FILTER} ${LIT_TESTS}/lsan/*/*.cc &

# DFSan
DFSAN_RTL=${COMPILER_RT}/lib/dfsan
run_lint ${DFSAN_RTL_LINT_FILTER} ${DFSAN_RTL}/*.cc \
                                  ${DFSAN_RTL}/*.h &
${DFSAN_RTL}/scripts/check_custom_wrappers.sh >> $ERROR_LOG

# Scudo
SCUDO_RTL=${COMPILER_RT}/lib/scudo
run_lint ${SCUDO_RTL_LINT_FILTER} ${SCUDO_RTL}/*.cpp \
                                  ${SCUDO_RTL}/*.h &

# Misc files
FILES=${COMMON_RTL}/*.inc
TMPFILES=""
for FILE in $FILES; do
  TMPFILE="$(${MKTEMP}).$(basename ${FILE}).cc"
  cp -f $FILE $TMPFILE
  run_lint ${COMMON_RTL_INC_LINT_FILTER} $TMPFILE &
  TMPFILES="$TMPFILES $TMPFILE"
done

wait

for temp in $TMPFILES; do
  rm -f $temp
done

if [ -s $ERROR_LOG ]; then
  cat $ERROR_LOG
  exit 1
fi

exit 0
