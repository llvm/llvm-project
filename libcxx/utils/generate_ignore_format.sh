#!/bin/bash

# Generate a list of headers that aren't formatted yet

if [ -z "${CLANG_FORMAT}" ]; then
  CLANG_FORMAT=clang-format
fi

rm libcxx/utils/data/ignore_format.txt
# This uses the same matches as the check-format CI step.
#
# Since it's hard to match empty extensions the following
# method is used, remove all files with an extension, then
# add the list of extensions that should be formatted.
for file in $(find libcxx/{benchmarks,include,src} -type f -not -name '*.*' -or \( \
	 -name "*.h" -or -name "*.hpp" -or \
	 -name "*.c" -or -name "*.cpp" -or \
	 -name "*.inc" -or -name "*.ipp" \
	 \) ); do

  ${CLANG_FORMAT} --Werror --dry-run ${file} >& /dev/null
  if [ $? != 0 ]; then
    echo ${file} >> libcxx/utils/data/ignore_format.txt
  fi
done

# Force sorting not to depend on the system's locale.
LC_ALL=C sort libcxx/utils/data/ignore_format.txt -d -o libcxx/utils/data/ignore_format.txt
