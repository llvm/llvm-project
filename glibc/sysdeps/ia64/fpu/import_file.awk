BEGIN {
	getline;
	while (!match($0, "^/[/*] static char cvs_id")) {
		print;
		getline;
	}
	getline;
	while (!match($0, "^// WARRANTY DISCLAIMER")) {
		print;
		if (!getline) {
			break;
		}
	}
	if (getline)
	{
		printf								      \
"// Redistribution and use in source and binary forms, with or without\n"     \
"// modification, are permitted provided that the following conditions are\n" \
"// met:\n"								      \
"//\n"									      \
"// * Redistributions of source code must retain the above copyright\n"	      \
"// notice, this list of conditions and the following disclaimer.\n"	      \
"//\n"									      \
"// * Redistributions in binary form must reproduce the above copyright\n"    \
"// notice, this list of conditions and the following disclaimer in the\n"    \
"// documentation and/or other materials provided with the distribution.\n"   \
"//\n"									      \
"// * The name of Intel Corporation may not be used to endorse or promote\n"  \
"// products derived from this software without specific prior written\n"     \
"// permission.\n\n";
		if (LICENSE_ONLY == "y") {
			do {
				print;
			} while (getline);
		}
	}
}

/^[.]data/ {
	print "RODATA";
	next;
}
/^([a-zA-Z_0-9]*_(tb[l0-9]|Tt|[tT]able|data|low|coeffs|constants|CONSTANTS|reduction|Stirling)(_?([1-9cdimpqstPQT]+|tail))?|(Constants|Poly|coeff)_.+|(double_sin_?cos|double_cis)[fl]?_.+):/ {
	table_name=substr($1,1,length($1)-1);
	printf "LOCAL_OBJECT_START(%s)\n", table_name;
	getline;
	while (!match($0, "^[ \t]*data")) {
		print;
		getline;
	}
	while (match($0, "(//|^[ \t]*data)")) {
		print;
		getline;
	}
	printf "LOCAL_OBJECT_END(%s)\n\n", table_name;
	next;
}
/^[.]proc[ \t]+__libm_(error_region|callout)/ {
	printf "LOCAL_LIBM_ENTRY(%s)\n", $2;
	getline;
	next;
}
/^[.]endp[ \t]+__libm_(error_region|callout)/ {
	printf "LOCAL_LIBM_END(%s)\n", $2;
	next;
}
/^[.]global/ {
	split($2, part, "#");
	name=part[1];
	if (match(name, "^"FUNC"$")) {
		next;
	}
}
/^[.]proc/ {
	split($2, part, "#");
	name=part[1];
	if (match(name, "^"FUNC"$")) {
		local_funcs=("^("			\
			     "cis|cisf|cisl"		\
			     "|cabs|cabsf|cabsl"	\
			     "|cot|cotf|cotl"		\
			     ")$");
		ieee754_funcs=("^("					  \
			       "atan2|atan2f|atan2l|atanl"		  \
			       "|cos|cosf|cosl"				  \
			       "|cosh|coshf|coshl"			  \
			       "|exp|expf|expl"				  \
			       "|exp10|exp10f|exp10l"			  \
			       "|expm1|expm1f|expm1l"			  \
			       "|fmod|fmodf|fmodl"			  \
			       "|hypot|hypotf|hypotl"			  \
			       "|fabs|fabsf|fabsl"			  \
			       "|floor|floorf|floorl"			  \
			       "|log1p|log1pf|log1pl"			  \
			       "|log|log10|log10f|log10l|log2l|logf|logl" \
			       "|remainder|remainderf|remainderl|"	  \
			       "|rint|rintf|rintl|"			  \
			       "|scalb|scalbf|scalbl"			  \
			       "|sin|sinf|sinl"				  \
			       "|sincos|sincosf|sincosl"		  \
			       "|sinh|sinhf|sinhl"			  \
			       "|sqrt|sqrtf|sqrtl"			  \
			       "|tan|tanf|tanl"				  \
			       ")$");
		if (match(name, ieee754_funcs)) {
			type="GLOBAL_IEEE754";
		} else if (match (name, local_funcs)) {
			type="LOCAL_LIBM";
		} else {
			type="GLOBAL_LIBM";
		}
		printf "%s_ENTRY(%s)\n", type, name;
		getline;
		while (!match($0, "^"name"#?:")) {
			getline;
		}
		getline;
		while (!match($0, "^.endp")) {
			print
			getline;
		}
		printf "%s_END(%s)\n", type, name;
		if (match(name, "^exp10[fl]?$")) {
			t=substr(name,6)
			printf "weak_alias (exp10%s, pow10%s)\n", t, t
		}
		next;
	}
}
/^[a-zA-Z_]+:/ {
	split($1, part, ":");
	name=part[1];
	if (match(name, "^"FUNC"$")) {
		printf "GLOBAL_LIBM_ENTRY(%s)\n", name;
		getline;
		while (!match($0, "^"name"#?:")) {
			getline;
		}
		getline;
		while (!match($0, "^.endp")) {
			print
			getline;
		}
		getline;
		printf "GLOBAL_LIBM_END(%s)\n", name;
		next;
	}
}

{ print }
