#!/bin/sh
# Generate test locale files.
# Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

common_objpfx="$1"; shift
localedef_before_env="$1"; shift
run_program_env="$1"; shift
localedef_after_env="$1"; shift
locfile="$1"; shift

generate_locale ()
{
  charmap=$1
  input=$2
  out=$3
  flags=$4
  ret=0
  ${localedef_before_env} ${run_program_env} I18NPATH=../localedata \
	${localedef_after_env} $flags -f $charmap -i $input \
	${common_objpfx}localedata/$out || ret=$?
  if [ $ret -eq 0 ]; then
    # The makefile checks the timestamp of the LC_CTYPE file,
    # but localedef won't have touched it if it was able to
    # hard-link it to an existing file.
    touch ${common_objpfx}localedata/$out/LC_CTYPE
  else
    echo "Charmap: \"${charmap}\" Inputfile: \"${input}\"" \
	 "Outputdir: \"${out}\" failed"
    exit 1
  fi
}

locfile=`echo $locfile|sed 's|.*/\([^/]*/LC_CTYPE\)|\1|'`
locale=`echo $locfile|sed 's|\([^.]*\)[.].*/LC_CTYPE|\1|'`
charmap=`echo $locfile|sed 's|[^.]*[.]\([^@ ]*\)\(@[^ ]*\)\?/LC_CTYPE|\1|'`
modifier=`echo $locfile|sed 's|[^.]*[.]\([^@ ]*\)\(@[^ ]*\)\?/LC_CTYPE|\2|'`

echo "Generating locale $locale.$charmap: this might take a while..."

# Run quietly and force output.
flags="--quiet -c"

# For SJIS the charmap is SHIFT_JIS. We just want the locale to have
# a slightly nicer name instead of using "*.SHIFT_SJIS", but that
# means we need a mapping here.
charmap_real="$charmap"
if [ "$charmap" = "SJIS" ]; then
  charmap_real="SHIFT_JIS"
fi

# In addition to this the SHIFT_JIS character maps are not ASCII
# compatible so we must use `--no-warnings=ascii' to disable the
# warning. See localedata/Makefile $(INSTALL-SUPPORTED-LOCALES)
# for the same logic.
if [ "$charmap_real" = 'SHIFT_JIS' ] \
   || [ "$charmap_real" = 'SHIFT_JISX0213' ]; then
  flags="$flags --no-warnings=ascii"
fi

generate_locale $charmap_real $locale$modifier $locale.$charmap$modifier "$flags"
