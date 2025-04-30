BEGIN { hv["0"] =  0; hv["1"] =  1; hv["2"] =  2; hv["3"] =  3;
	hv["4"] =  4; hv["5"] =  5; hv["6"] =  6; hv["7"] =  7;
	hv["8"] =  8; hv["9"] =  9; hv["A"] = 10; hv["B"] = 11;
	hv["C"] = 12; hv["D"] = 13; hv["E"] = 14; hv["F"] = 15;
	hv["a"] = 10; hv["b"] = 11; hv["c"] = 12; hv["d"] = 13;
	hv["e"] = 14; hv["f"] = 15;

	first = 0; last = 0; idx = 0; f = 0;
}

function tonum(str)
{
  num=0;
  cnt=1;
  while (cnt <= length(str)) {
    num *= 16;
    num += hv[substr(str,cnt,1)];
    ++cnt;
  }
  return num;
}

function fmt(val)
{
  if (f++ % 8 == 0)
    { printf ("\n  '\\x%02x',", val); }
  else
    { printf (" '\\x%02x',", val); }
}

{
  u = tonum($1); c = tonum($2);

  if (u - last > 6)
    {
      if (last) { idx += last - first + 1; }
      first = u;
    }
  else
    {
      for (m = last+1; m < u; m++) { fmt(0); }
    }

  fmt(c);
  last = u;
}

END { print "" }
