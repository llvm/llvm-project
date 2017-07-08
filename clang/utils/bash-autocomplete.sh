# Please add "source /path/to/bash-autocomplete.sh" to your .bashrc to use this.

_clang_filedir()
{
  # _filedir function provided by recent versions of bash-completion package is
  # better than "compgen -f" because the former honors spaces in pathnames while
  # the latter doesn't. So we use compgen only when _filedir is not provided.
  _filedir 2> /dev/null || COMPREPLY=( $( compgen -f ) )
}

_clang()
{
  local cur prev words cword arg flags w1 w2
  # If latest bash-completion is not supported just initialize COMPREPLY and
  # initialize variables by setting manualy.
  _init_completion -n 2> /dev/null
  if [[ "$?" != 0 ]]; then
    COMPREPLY=()
    cword=$COMP_CWORD
    cur="${COMP_WORDS[$cword]}"
  fi

  # bash always separates '=' as a token even if there's no space before/after '='.
  # On the other hand, '=' is just a regular character for clang options that
  # contain '='. For example, "-stdlib=" is defined as is, instead of "-stdlib" and "=".
  # So, we need to partially undo bash tokenization here for integrity.
  w1="${COMP_WORDS[$cword - 1]}"
  if [[ $cword > 1 ]]; then
    w2="${COMP_WORDS[$cword - 2]}"
  # Clang want to know if -cc1 or -Xclang option is specified or not, because we don't want to show
  # cc1 options otherwise.
  if [[ "${COMP_WORDS[1]}" == "-cc1" || "$w1" == "-Xclang" ]]; then
    arg="#"
  fi
  if [[ "$cur" == -* ]]; then
    # -foo<tab>
    arg="$arg$cur"
  elif [[ "$w1" == -*  && "$cur" == '=' ]]; then
    # -foo=<tab>
    arg="$arg$w1=,"
  elif [[ "$cur" == -*= ]]; then
    # -foo=<tab>
    arg="$arg$cur,"
  elif [[ "$w1" == -* ]]; then
    # -foo <tab> or -foo bar<tab>
    arg="$arg$w1,$cur"
  elif [[ "$w2" == -* && "$w1" == '=' ]]; then
    # -foo=bar<tab>
    arg="$arg$w2=,$cur"
  elif [[ ${cur: -1} != '=' && ${cur/=} != $cur ]]; then
    # -foo=bar<tab>
    arg="$arg${cur%=*}=,${cur#*=}"
  fi

  # expand ~ to $HOME
  eval local path=${COMP_WORDS[0]}
  flags=$( "$path" --autocomplete="$arg" 2>/dev/null )
  # If clang is old that it does not support --autocomplete,
  # fall back to the filename completion.
  if [[ "$?" != 0 ]]; then
    _clang_filedir
    return
  fi

  if [[ "$cur" == '=' ]]; then
    COMPREPLY=( $( compgen -W "$flags" -- "") )
  elif [[ "$flags" == "" || "$arg" == "" ]]; then
    _clang_filedir
  else
    # Bash automatically appends a space after '=' by default.
    # Disable it so that it works nicely for options in the form of -foo=bar.
    [[ "${flags: -1}" == '=' ]] && compopt -o nospace 2> /dev/null
    COMPREPLY=( $( compgen -W "$flags" -- "$cur" ) )
  fi
}
complete -F _clang clang
