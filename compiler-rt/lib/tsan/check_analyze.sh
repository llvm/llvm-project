#!/bin/bash
#
# Script that checks that critical functions in TSan runtime have correct number
# of push/pop/rsp instructions to verify that runtime is efficient enough.

set -u

if [[ "$#" != 1 ]]; then
  echo "Usage: $0 /path/to/binary/built/with/tsan"
  exit 1
fi

SCRIPTDIR=$(dirname $0)
RES=$(${SCRIPTDIR}/analyze_libtsan.sh $1)
PrintRes() {
  printf "%s\n" "$RES"
}

PrintRes

check() {
  res=$(PrintRes | egrep "$1 .* $2 $3; ")
  if [ "$res" == "" ]; then
    echo FAILED $1 must contain $2 $3
    exit 1
  fi
}

for f in write1 write2 write4 write8; do
  check $f rsp 1
  check $f push 2
  check $f pop 12
done

for f in read1 read2 read4 read8; do
  check $f rsp 1
  check $f push 3
  check $f pop 3
done

for f in func_entry func_exit; do
  check $f rsp 0
  check $f push 0
  check $f pop 0
  check $f call 1  # TraceSwitch()
done

echo LGTM
