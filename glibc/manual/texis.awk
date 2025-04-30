BEGIN {
    print "texis = \\";
    for(x = 1; x < ARGC; x++)
    {
	input[0] = ARGV[x];
	print ARGV[x], "\\";
	for (s = 0; s >= 0; s--)
	{
	    while ((getline < input[s]) > 0)
	    {
		if ($1 == "@include")
		{
		    input[++s] = $2;
		    print $2, "\\";
		}
	    }
	    close(input[s]);
	}
    }
    print "";
}
