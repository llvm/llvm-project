#!/usr/local/bin/gawk -f
BEGIN {
  last_node="";
}

/^@node/ {
  name = $0;
  sub(/^@node +/, "", name);
  sub(/[@,].*$/, "", name);
  last_node = name;
}

/^@deftype(fn|vr)/ {
# The string we want is $4, except that if there were brace blocks
# before that point then it gets shifted to the right, since awk
# doesn't know from brace blocks.
  id = 4; check = 2; squig = 0;
  while(check < id)
  {
    if($check ~ /{/) squig++;
    if($check ~ /}/) squig--;
    if(squig) id++;
    check++;
  }

  gsub(/[(){}*]/, "", $id);
  printf ("* %s: (libc)%s.\n", $id, last_node);
}

/^@deftypefun/ {
# Likewise, except it's $3 theoretically.
  id = 3; check = 2; squig = 0;
  while(check < id)
  {
    if($check ~ /{/) squig++;
    if($check ~ /}/) squig--;
    if(squig) id++;
    check++;
  }

  gsub(/[(){}*]/, "", $id);
  printf ("* %s: (libc)%s.\n", $id, last_node);
}
