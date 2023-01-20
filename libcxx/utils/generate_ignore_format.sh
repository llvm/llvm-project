#!/bin/bash

# Generate a list of headers that aren't formatted yet

if [ -z "${CLANG_FORMAT}" ]; then
  CLANG_FORMAT=clang-format
fi

rm libcxx/utils/data/ignore_format.txt
for file in $(find libcxx/{benchmarks,include,src}/ -type f); do
  ${CLANG_FORMAT} --Werror --dry-run ${file} >& /dev/null
  if [ $? != 0 ]; then
    echo ${file} >> libcxx/utils/data/ignore_format.txt
  fi
done

sort libcxx/utils/data/ignore_format.txt -d -o libcxx/utils/data/ignore_format.txt
