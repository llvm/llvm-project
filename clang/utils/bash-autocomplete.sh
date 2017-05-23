# Please add "source /path/to/bash-autocomplete.sh" to your .bashrc to use this.
_clang()
{
  local cur prev words cword flags
  _init_completion -n : || return

  flags=$( clang --autocomplete="$cur" )
  if [[ "$flags" == "" || "$cur" == "" ]]; then
    _filedir
  else
    COMPREPLY=( $( compgen -W "$flags" -- "$cur" ) )
  fi
}
complete -F _clang clang
