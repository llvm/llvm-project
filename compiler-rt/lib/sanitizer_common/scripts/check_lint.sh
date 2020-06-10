#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${COMPILER_RT}" = "" ]; then
  COMPILER_RT=$(readlink -f $SCRIPT_DIR/../../..)
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
-build/namespaces,-build/c++11,-runtime/int

COMMON_LIT_TEST_LINT_FILTER=-whitespace/indent,-whitespace/line_length,-runtime/arrays,-readability/braces

ASAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
ASAN_TEST_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/printf,-runtime/threadsafe_fn
ASAN_LIT_TEST_LINT_FILTER=${ASAN_TEST_LINT_FILTER},${COMMON_LIT_TEST_LINT_FILTER}

TSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER},-readability/braces
TSAN_TEST_LINT_FILTER=${TSAN_RTL_LINT_FILTER},-runtime/threadsafe_fn
TSAN_LIT_TEST_LINT_FILTER=${TSAN_TEST_LINT_FILTER},${COMMON_LIT_TEST_LINT_FILTER}

MSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}

LSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
LSAN_LIT_TEST_LINT_FILTER=${LSAN_RTL_LINT_FILTER},${COMMON_LIT_TEST_LINT_FILTER}

DFSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
SCUDO_RTL_LINT_FILTER=${COMMON_LINT_FILTER}

COMMON_RTL_INC_LINT_FILTER=${COMMON_LINT_FILTER}

SANITIZER_INCLUDES_LINT_FILTER=${COMMON_LINT_FILTER}

MKTEMP_DIR=$(mktemp -qd /tmp/check_lint.XXXXXXXXXX)
MKTEMP="mktemp -q ${MKTEMP_DIR}/tmp.XXXXXXXXXX"
cleanup() {
  rm -rf $MKTEMP_DIR
}
trap cleanup EXIT

EXITSTATUS=0
ERROR_LOG=$(${MKTEMP})

run_lint() {
  FILTER=$1
  shift
  TASK_LOG=$(${MKTEMP})
  ${CPPLINT} --filter=${FILTER} "$@" > $TASK_LOG 2>&1
  if [ "$?" != "0" ]; then
    cat $TASK_LOG | grep -v "Done processing" | grep -v "Total errors found" \
      | grep -v "Skipping input" >> $ERROR_LOG
  fi
  if [ "${SILENT}" != "1" ]; then
    cat $TASK_LOG
  fi
  ${LITLINT} "$@" 2>>$ERROR_LOG
}

LIT_TESTS=${COMPILER_RT}/test
# Headers
SANITIZER_INCLUDES=${COMPILER_RT}/include/sanitizer
FUZZER_INCLUDES=${COMPILER_RT}/include/fuzzer
run_lint ${SANITIZER_INCLUDES_LINT_FILTER} ${SANITIZER_INCLUDES}/*.h \
                                           ${FUZZER_INCLUDES}/*.h &

# Sanitizer_common
COMMON_RTL=${COMPILER_RT}/lib/sanitizer_common
run_lint ${COMMON_RTL_INC_LINT_FILTER} ${COMMON_RTL}/*.cpp \
                                       ${COMMON_RTL}/*.h \
                                       ${COMMON_RTL}/tests/*.cpp &

# Interception
INTERCEPTION=${COMPILER_RT}/lib/interception
run_lint ${ASAN_RTL_LINT_FILTER} ${INTERCEPTION}/*.cpp \
                                 ${INTERCEPTION}/*.h &

# ASan
ASAN_RTL=${COMPILER_RT}/lib/asan
run_lint ${ASAN_RTL_LINT_FILTER} ${ASAN_RTL}/*.cpp \
                                 ${ASAN_RTL}/*.h &
run_lint ${ASAN_TEST_LINT_FILTER} ${ASAN_RTL}/tests/*.cpp \
                                  ${ASAN_RTL}/tests/*.h &
run_lint ${ASAN_LIT_TEST_LINT_FILTER} ${LIT_TESTS}/asan/*/*.cpp &

# TSan
TSAN_RTL=${COMPILER_RT}/lib/tsan
run_lint ${TSAN_RTL_LINT_FILTER} ${TSAN_RTL}/rtl/*.cpp \
                                 ${TSAN_RTL}/rtl/*.h &
run_lint ${TSAN_TEST_LINT_FILTER} ${TSAN_RTL}/tests/rtl/*.cpp \
                                  ${TSAN_RTL}/tests/rtl/*.h \
                                  ${TSAN_RTL}/tests/unit/*.cpp &
run_lint ${TSAN_LIT_TEST_LINT_FILTER} ${LIT_TESTS}/tsan/*.cpp &

# MSan
MSAN_RTL=${COMPILER_RT}/lib/msan
run_lint ${MSAN_RTL_LINT_FILTER} ${MSAN_RTL}/*.cpp \
                                 ${MSAN_RTL}/*.h &

# LSan
LSAN_RTL=${COMPILER_RT}/lib/lsan
run_lint ${LSAN_RTL_LINT_FILTER} ${LSAN_RTL}/*.cpp \
                                 ${LSAN_RTL}/*.h &
run_lint ${LSAN_LIT_TEST_LINT_FILTER} ${LIT_TESTS}/lsan/*/*.cpp &

# DFSan
DFSAN_RTL=${COMPILER_RT}/lib/dfsan
run_lint ${DFSAN_RTL_LINT_FILTER} ${DFSAN_RTL}/*.cpp \
                                  ${DFSAN_RTL}/*.h &
${DFSAN_RTL}/scripts/check_custom_wrappers.sh >> $ERROR_LOG

# Scudo
SCUDO_RTL=${COMPILER_RT}/lib/scudo
run_lint ${SCUDO_RTL_LINT_FILTER} ${SCUDO_RTL}/*.cpp \
                                  ${SCUDO_RTL}/*.h &

# Misc files
(
rsync -a --prune-empty-dirs --exclude='*/profile/*' --exclude='*/builtins/*' --exclude='*/xray/*' --include='*/' --include='*.inc' --exclude='*' "${COMPILER_RT}/" "${MKTEMP_DIR}/"
find ${MKTEMP_DIR} -type f -name '*.inc' -exec mv {} {}.cpp \;
( ERROR_LOG=${ERROR_LOG}.inc run_lint ${COMMON_RTL_INC_LINT_FILTER} $(find ${MKTEMP_DIR} -type f -name '*.inc.cpp') )
sed "s|${MKTEMP_DIR}|${COMPILER_RT}|g" ${ERROR_LOG}.inc | sed "s|.inc.cpp|.inc|g" >> ${ERROR_LOG}
) &

wait

if [ -s $ERROR_LOG ]; then
  cat $ERROR_LOG
  exit 1
fi

exit 0
