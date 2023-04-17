# Usage: set-release-binary-outputs.sh <github_user> <tag> <upload>

set -e

if [ -z "$GITHUB_OUTPUT" ]; then
  export GITHUB_OUTPUT=`mktemp`
  echo "Warning: Environment variable GITHUB_OUTPUT is not set."
  echo "Writing output variables to $GITHUB_OUTPUT"
fi

github_user=$1
tag=$2
upload=$3

if [[ "$github_user" != "tstellar" && "$github_user" != "tru" ]]; then
  echo "ERROR: User not allowed: $github_user"
  exit 1
fi
pattern='^llvmorg-[0-9]\+\.[0-9]\+\.[0-9]\+\(-rc[0-9]\+\)\?$'
echo "$tag" | grep -e $pattern
if [ $? != 0 ]; then
  echo "ERROR: Tag '$tag' doesn't match pattern: $pattern"
  exit 1
fi
release_version=`echo "$tag" | sed 's/llvmorg-//g'`
release=`echo "$release_version" | sed 's/-.*//g'`
build_dir=`echo "$release_version" | sed 's,^[^-]\+,final,' | sed 's,[^-]\+-rc\(.\+\),rc\1,'`
rc_flags=`echo "$release_version" | sed 's,^[^-]\+,-final,' | sed 's,[^-]\+-rc\(.\+\),-rc \1 -test-asserts,' | sed 's,--,-,'`
echo "release-version=$release_version" >> $GITHUB_OUTPUT
echo "release=$release" >> $GITHUB_OUTPUT
echo "build-dir=$build_dir" >> $GITHUB_OUTPUT
echo "rc-flags=$rc_flags" >> $GITHUB_OUTPUT
echo "upload=$upload" >> $GITHUB_OUTPUT
echo "ref=$tag" >> $GITHUB_OUTPUT
