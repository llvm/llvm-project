/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


/* C counterpart to test Fortran BIND(C) functions, 
   subroutines
 */

#ifdef __cplusplus
extern "C" void printf (char * , ...);
#else
extern  void printf (char * , ...);
#endif

#define N 28
#define ND 4


extern int expect[N]= {  /* newc */ 
			0, 1, 2, 4,
			/* intfunc */
			3, 4, 5, 6, 10, 20, 30, 40, 45,
			/* logfunc */
			7, 8, 9, 10, -1, 0, -1, 0,
			/* real func */
			2.0, 3.0, 4.0,      22 ,        3.0,
			3.0, 6.0 };
			
extern  double d_expect[ND]= { /* realfunc */
			 0.0, 300300.0, 4000.04, 50.0};
int result[N];
double d_result[ND];



void newc (int *a, int *b, float * kk) {


result[0] = 0;
result[1] = *a;
result[2] = *b;
result[3] = *kk;

}


void intfunc( int a, int b, int c, int d, char i1, 
		     short i2, int i3, int i4, long long i8) {

result[4] = a;
result[5] = b;
result[6] = c;
result[7] = d;
result[8] = (int) i1;
result[9] = (int) i2;
result[10] = i3;
result[11] = i4;
result[12] = (int) i8;

}

void  logfunc( int a, int b, int c, int d, 
		     int i2, int i3, int i4, int i8) {
result[13] = a;
result[14] = b;
result[15] = c;
result[16] = d;
result[17] = i2;
result[18] = i3;
result[19] = i4;
result[20] = i8;

}

int  realfunc( float a, float b, float c, double d, 
		     float e, float f, float g,
		     float r4, double r8) {

result[21] = a;
result[22] = b;
result[23] = c;
result[24] = 0; /* f90 will store return val */
result[25] = e;
result[26] = f;
result[27] = g;
d_result[0] = 0.0;
d_result[1] = r4;
d_result[2] = r8;
d_result[3] = d;
return (22);

}
 

