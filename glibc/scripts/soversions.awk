# awk script for shlib-versions.v -> soversions.i; see Makeconfig.

# Obey the first matching DEFAULT line.
$1 == "DEFAULT" {
  $1 = "";
  default_set[++ndefault_set] = $0;
  next
}

# Collect all lib lines before emitting anything, so DEFAULT
# can be interspersed.
{
  lib = number = $1;
  sub(/=.*$/, "", lib);
  sub(/^.*=/, "", number);
  if (lib in numbers) next;
  numbers[lib] = number;
  order[lib] = ++order_n;
  if (NF > 1) {
    $1 = "";
    versions[lib] = $0
  }
}

END {
  for (lib in numbers) {
    if (lib in versions)
      set = versions[lib];
    else {
      set = "";
      if (ndefault_set >= 1)
	set = default_set[1];
    }
    line = set ? (lib FS numbers[lib] FS set) : (lib FS numbers[lib]);
    if (!(lib in lineorder) || order[lib] < lineorder[lib]) {
      lineorder[lib] = order[lib];
      lines[lib] = "DEFAULT" FS line;
    }
  }
  for (c in lines) {
    print lines[c]
  }
}
