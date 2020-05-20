# Common functionality for test scripts
# Process arguments, expecting source file as 1st; optional path to f18 as 2nd
# Set: $F18 to the path to f18 with options; $temp to an empty temp directory;
# and $src to the full path of the single source argument.

function die {
  echo "$(basename $0): $*" >&2
  exit 1
}
if [[ $# < 3 ]]; then
  echo "Usage: $(basename $0) <fortran-source> <temp test dir> <f18-command>"
  exit 1
fi

case $1 in
  (/*) src="$1" ;;
  (*) src="$(dirname $0)/$1" ;;
esac
shift
temp=$1
mkdir -p $temp
shift

[[ ! -f $1 ]] && die "f18 executable not found: $1"
F18="$*"
