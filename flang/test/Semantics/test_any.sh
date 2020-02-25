#!/usr/bin/env bash
# Compile a source file with '-funparse-with-symbols' and verify
# we get the right symbols in the output, i.e. the output should be
# the same as the input, except for the copyright comment.
# Change the compiler by setting the F18 environment variable.

srcdir=$(dirname $0)
source $srcdir/common.sh

FileCheck=${FileCheck:=internal_check}

function internal_check() {
  r=true
  linput="$1"
  lstdin=`mktemp`
  lcheck=`mktemp`
  cat - > ${lstdin}
  egrep '^[[:space:]]*![[:space:]]*CHECK:[[:space:]]*' ${linput} | sed -e 's/^[[:space:]]*![[:space:]]*CHECK:[[:space:]]*//' > ${lcheck} 2>/dev/null
  while read p; do
    if egrep "${p}" ${lstdin} >/dev/null 2>&1; then
      true
    else
      echo "Not found: ${p}" >&2
      r=false
    fi
  done < ${lcheck}
  egrep '^[[:space:]]*![[:space:]]*CHECK-NOT:[[:space:]]*' ${linput} | sed -e 's/^[[:space:]]*![[:space:]]*CHECK-NOT:[[:space:]]*//' > ${lcheck} 2>/dev/null
  while read p; do
    if egrep "${p}" ${lstdin} >/dev/null 2>&1; then
      echo "Found: ${p}" >&2
      r=false
    fi
  done < ${lcheck}
  egrep '^[[:space:]]*![[:space:]]*CHECK-ONCE:[[:space:]]*' ${linput} | sed -e 's/^[[:space:]]*![[:space:]]*CHECK-ONCE:[[:space:]]*//' > ${lcheck} 2>/dev/null
  while read p; do
    count=$(egrep -o -e "${p}" ${lstdin} | wc -l)
    if [ ${count} -eq 0 ]; then
      echo "Not found: ${p}" >&2
      r=false
    fi
    if [ ${count} -gt 1 ]; then
      echo "Found duplicates: ${p}" >&2
      r=false
    fi
  done < ${lcheck}
  rm -f ${lstdin} ${lcheck}
  ${r}
}

gr=0
for input in ${srcdir}/$*; do
  [[ ! -f $input ]] && die "File not found: $input"
  CMD=$(cat ${input} | egrep '^[[:space:]]*![[:space:]]*RUN:[[:space:]]*' | sed -e 's/^[[:space:]]*![[:space:]]*RUN:[[:space:]]*//')
  CMD=$(echo ${CMD} | sed -e "s:%s:${input}:g")
  if egrep -q -e '%t' <<< ${CMD} ; then
    CMD=$(echo ${CMD} | sed -e "s:%t:$temp/t:g")
  fi
  if $(eval $CMD); then
    echo "PASS  ${input}"
  else
    echo "FAIL  ${input}"
    gr=1
  fi
done
exit $gr
