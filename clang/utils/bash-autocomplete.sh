# Please add "source /path/to/bash-autocomplete.sh" to your .bashrc to use this.
_clang()
{
  local cur prev words cword arg
  _init_completion -n : || return

  # bash always separates '=' as a token even if there's no space before/after '='.
  # On the other hand, '=' is just a regular character for clang options that
  # contain '='. For example, "-stdlib=" is defined as is, instead of "-stdlib" and "=".
  # So, we need to partially undo bash tokenization here for integrity.
  local w1="${COMP_WORDS[$cword - 1]}"
  local w2="${COMP_WORDS[$cword - 2]}"
  if [[ "$cur" == -* ]]; then
    # -foo<tab>
    arg="$cur"
  elif [[ "$w1" == -*  && "$cur" == '=' ]]; then
    # -foo=<tab>
    arg="$w1=,"
  elif [[ "$w1" == -* ]]; then
    # -foo <tab> or -foo bar<tab>
    arg="$w1,$cur"
  elif [[ "$w2" == -* && "$w1" == '=' ]]; then
    # -foo=bar<tab>
    arg="$w2=,$cur"
  else
    _filedir
  fi

  local flags=$( clang --autocomplete="$arg" )
  if [[ "$cur" == "=" ]]; then
    COMPREPLY=( $( compgen -W "$flags" -- "") )
  elif [[ "$flags" == "" ]]; then
    _filedir
  else
    COMPREPLY=( $( compgen -W "$flags" -- "$cur" ) )
  fi
}
complete -F _clang clang
