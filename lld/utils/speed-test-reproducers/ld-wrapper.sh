#! @shell@

source @out@/nix-support/utils.bash

expandResponseParams "$@"

output="a.out"
should_add_repro=true
newparams=()
for arg in "${params[@]}"; do
  case "$arg" in
    -r|--version)
      should_add_repro=false
      ;;
    *)
      ;;
  esac
  case "$prev" in
    -o)
      output="$arg"
      newparams+=("$arg")
      ;;
    *)
      if [ -e "$arg.nolldrepro" ]; then
        newparams+=("$arg.nolldrepro")
      else
        newparams+=("$arg")
      fi
      ;;
  esac
  prev="$arg"
done

export LLD_REPRODUCE="$output.repro.tar"
if @targetPrefix@nix-wrap-lld "${newparams[@]}"; then
  if $should_add_repro; then
    @lz4@ -c "$LLD_REPRODUCE" > "$LLD_REPRODUCE.lz4"
    mv "$output" "$output.nolldrepro"
    @targetPrefix@objcopy --add-section ".lld_repro=$LLD_REPRODUCE.lz4" "$output.nolldrepro" "$output"
    rm -f "$LLD_REPRODUCE.lz4"
  fi
  exitcode=0
else
  # Some Nix packages don't link with lld so just use bfd instead.
  @targetPrefix@ld.bfd "${newparams[@]}"
  exitcode=$?
fi

rm -f "$LLD_REPRODUCE"
exit $exitcode
