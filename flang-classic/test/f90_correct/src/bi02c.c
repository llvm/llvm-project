/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/* C counterpart to test Fortran parameter passing of C_LOC()
   and TYPE(C_PTR) 
 */


#define N 77
#define ND 4


extern int expect[N]= {  3, 55, 11, 33,  3, 12, 12, 12, 12, 3, 
			 3, 3,  3,  3,   3,  3,  3,  3,  3, 3,
			 3, 3,  3, 12,  12, 12, 12,  3,  3, 3, 
			 3, 3,  3,  3,   3,  3,  3,  3,  3, 3,
			 3, 3,  3,  3,   3,  3,  3,  3,  3, 3,
			 3, 12, 55, 11,  5,  3, 12,  5,  5, 3,
			12, 5,  33, 3,  12, 12, 12, 12,  3, 3,
			 5, 3,  12, 5, 3, 3, 3
		      };
extern int result[N]= { 0};

int c= 0;

extern void cp_call (int i, int * j, int *k, int ll) { 
	result[c++] = i;
	result[c++] = *j;
	result[c++] = *k;
	result[c++] = ll;
}
extern void c_call_ref (int ** ii) {
	result[c++] = **ii;
}

extern void c_call (int * i) { 
	result[c++] = *i;
}
extern int * c_fun (int * i) { 

	result[c++] = *i;
	return i;
}
extern int * c_fun_ptr (int ** i) { 
	result[c++] = **i;
	return *i;
}
extern int * c_fun_ref (int ** i) { 
	result[c++] = **i;
	return *i ;
}
extern int * cp_fun (int i, int * j, int *k, int i2) { 
	result[c++] = i;
	result[c++] = *j;
	result[c++] = *k;
	result[c++] = i2;
	return j;
}

extern int * get_ptr (int * pp) { 

	result[c++] = *pp;
        return pp; 
}

extern int * get_ptr_ref ( int ** pp2) {
	result[c++] = **pp2;
	return*  pp2; 
} 


