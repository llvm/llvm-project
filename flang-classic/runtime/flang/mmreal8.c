/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* mmreal8.c -- F90 fast-/dgemm-like MATMUL intrinsics for real*8 type */

#include "stdioInterf.h"
#include "fioMacros.h"

#define SMALL_ROWSA 10
#define SMALL_ROWSB 10
#define SMALL_COLSB 10

/* C prototypes for external Fortran functions */
void ftn_mvmul_real8_(int *, __POINT_T *, __POINT_T *, double *, double *,
                      __POINT_T *, double *, double *, double *);
void ftn_vmmul_real8_(int *, __POINT_T *, __POINT_T *, double *, double *,
                      double *, __POINT_T *, double *, double *);
void ftn_mnaxnb_real8_(__POINT_T *, __POINT_T *, __POINT_T *, double *,
                       double *, __POINT_T *, double *, __POINT_T *,
                       double *, double *, __POINT_T *);
void ftn_mnaxtb_real8_(__POINT_T *, __POINT_T *, __POINT_T *, double *,
                       double *, __POINT_T *, double *, __POINT_T *,
                       double *, double *, __POINT_T *);
void ftn_mtaxnb_real8_(__POINT_T *, __POINT_T *, __POINT_T *, double *,
                       double *, __POINT_T *, double *, __POINT_T *,
                       double *, double *, __POINT_T *);
void ftn_mtaxtb_real8_(__POINT_T *, __POINT_T *, __POINT_T *, double *,
                       double *, __POINT_T *, double *, __POINT_T *,
                       double *, double *, __POINT_T *);

void ENTF90(MMUL_REAL8, mmul_real8)(int ta, int tb, __POINT_T mra,
                                    __POINT_T ncb, __POINT_T kab, double *alpha,
                                    double a[], __POINT_T lda, double b[],
                                    __POINT_T ldb, double *beta, double c[],
                                    __POINT_T ldc)

{
  /*
  *   Notes on parameters
  *   ta = 0 -> no transpose of matrix a
  *   tb = 0 -> no transpose of matrix b

  *   mra = number of rows in matrices a and c ( = m )
  *   ncb = number of columns in matrices b and c ( = n )
  *   kab = shared dimension of matrices a and b ( = k, but need k elsewhere )
  *   a = starting address of matrix a
  *   b = starting address of matrix b
  *   c = starting address of matric c
  *   lda = leading dimension of matrix a
  *   ldb = leading dimension of matrix b
  *   ldc = leading dimension of matrix c
  *   alpha = 1.0
  *   beta = 0.0
  *   Note that these last two conditions are inconsitent with the general
  *   case for dgemm.
  *   Taken together we have
  *   c = beta * c + alpha * ( (ta)a * (tb)*b )
  *   where the meaning of (ta) and (tb) is that if ta = 0 a is not transposed
  *   and transposed otherwise and if tb = 0, b is not transpose and transposed
  *   otherwise.
  */

  // Local variables

  int colsa, rowsa, rowsb, colsb;
  int tindex = 0;
  double buffera[SMALL_ROWSA * SMALL_ROWSB];
  double bufferb[SMALL_COLSB * SMALL_ROWSB];
  double temp;
  double calpha, cbeta;
  /*
   * Small matrix multiply variables
   */
  int i, ja, j, k;
  int astrt = 0, bstrt, cstrt, andx, bndx, cndx, indx, indx_strt;
  /*
   * tindex has the following meaning:
   * ta == 0, tb == 0: tindex = 0
   * ta == 1, tb == 0: tindex = 1
   * ta == 0, tb == 1; tindex = 2
   * ta == 1, tb == 1; tindex = 3
   */
  /*  if( ( tb == 0 ) && ( ncb == 1 ) && ( ldb == 1 ) ){ */
  if ((tb == 0) && (ncb == 1)) {
    /* matrix vector multiply */
    ftn_mvmul_real8_(&ta, &mra, &kab, alpha, a, &lda, b, beta, c);
    return;
  }
  if ((ta == 0) && (mra == 1) && (ldc == 1)) {
    /* vector matrix multiply */
    ftn_vmmul_real8_(&tb, &ncb, &kab, alpha, a, b, &ldb, beta, c);
    return;
  }

  calpha = *alpha;
  cbeta = *beta;
  rowsa = mra;
  colsa = kab;
  rowsb = kab;
  colsb = ncb;
  if (ta == 1)
    tindex = 1;

  if (tb == 1)
    tindex += 2;

  // Check for really small matrix sizes

  // Check for really small matrix sizes

  if ((colsb <= SMALL_COLSB) && (rowsa <= SMALL_ROWSA) &&
      (rowsb <= SMALL_ROWSB)) {
    switch (tindex) {
    case 0: /* matrix a and matrix b normally oriented
             *
             * The notation here refers to the Fortran orientation since
             * that is the origination of these matrices
             */
      astrt = 0;
      bstrt = 0;
      cstrt = 0;
      if (cbeta == (double)0.0) {
        for (i = 0; i < rowsa; i++) {
          /* Transpose the a row of the a matrix */
          andx = astrt;
          indx = 0;
          for (ja = 0; ja < colsa; ja++) {
            buffera[indx++] = calpha * a[andx];
            andx += lda;
          }
          astrt++;
          cndx = cstrt;
          for (j = 0; j < colsb; j++) {
            temp = 0.0;
            bndx = bstrt;
            for (k = 0; k < rowsb; k++)
              temp += buffera[k] * b[bndx++];
            bstrt += ldb;
            c[cndx] = temp;
            cndx += ldc;
          }
          cstrt++; /* set index for next row of c */
          bstrt = 0;
        }
      } else {
        for (i = 0; i < rowsa; i++) {
          /* Transpose the a row of the a matrix */
          andx = astrt;
          indx = 0;
          for (ja = 0; ja < colsa; ja++) {
            buffera[indx++] = calpha * a[andx];
            andx += lda;
          }
          astrt++;
          cndx = cstrt;
          for (j = 0; j < colsb; j++) {
            temp = 0.0;
            bndx = bstrt;
            for (k = 0; k < rowsb; k++)
              temp += buffera[k] * b[bndx++];
            bstrt += ldb;
            c[cndx] = temp + cbeta * c[cndx];
            cndx += ldc;
          }
          cstrt++; /* set index for next row of c */
          bstrt = 0;
        }
      }

      break;
    case 1: /* matrix a transpose, matrix b normally oriented */
      bndx = 0;
      cstrt = 0;
      andx = 0;
      if (cbeta == (double)0.0) {
        for (j = 0; j < colsb; j++) {
          cndx = cstrt;
          for (i = 0; i < rowsa; i++) {
            /* Matrix a need not be transposed */
            temp = 0.0;
            for (k = 0; k < rowsb; k++)
              temp += a[andx + k] * b[bndx + k];
            c[cndx] = calpha * temp;
            andx += lda;
            cndx++;
          }
          cstrt += ldc; /* set index for next column of c */
          astrt++;      /* set index for next column of a */
          b += ldb;
          andx = 0;
        }
      } else {
        for (j = 0; j < colsb; j++) {
          cndx = cstrt;
          for (i = 0; i < rowsa; i++) {
            /* Matrix a need not be transposed */
            temp = 0.0;
            for (k = 0; k < rowsb; k++)
              temp += a[andx + k] * b[bndx + k];
            c[cndx] = calpha * temp + cbeta * c[cndx];
            andx += lda;
            cndx++;
          }
          cstrt += ldc; /* set index for next column of c */
          astrt++;      /* set index for next column of a */
          b += ldb;
          andx = 0;
        }
      }

      break;
    case 2: /* Matrix a normal, b transposed */
      /* We will transpose b and work with transposed rows of a */
      /* Transpose matrix b */
      indx_strt = 0;
      bstrt = 0;
      for (j = 0; j < rowsb; j++) {
        indx = indx_strt;
        bndx = bstrt;
        for (i = 0; i < colsb; i++) {
          bufferb[indx] = calpha * b[bndx++];
          indx += rowsb;
        }
        indx_strt++;
        bstrt += ldb;
      }
      /* All of b is now transposed */

      astrt = 0;
      cstrt = 0;
      if (cbeta == (double)0.0) {
        for (i = 0; i < rowsa; i++) {
          /* Transpose the a row of the a matrix */
          andx = astrt;
          indx = 0;
          for (ja = 0; ja < colsa; ja++) {
            buffera[indx++] = a[andx];
            andx += lda;
          }
          cndx = cstrt;
          bndx = 0;
          for (j = 0; j < colsb; j++) {
            temp = 0.0;
            for (k = 0; k < rowsb; k++)
              temp += buffera[k] * bufferb[bndx++];
            c[cndx] = temp;
            cndx += ldc;
          }
          cstrt++; /* set index for next row of c */
          astrt++;
        }
      } else {
        for (i = 0; i < rowsa; i++) {
          /* Transpose the a row of the a matrix */
          andx = astrt;
          indx = 0;
          for (ja = 0; ja < colsa; ja++) {
            buffera[indx++] = a[andx];
            andx += lda;
          }
          cndx = cstrt;
          bndx = 0;
          for (j = 0; j < colsb; j++) {
            temp = 0.0;
            for (k = 0; k < rowsb; k++)
              temp += buffera[k] * bufferb[bndx++];
            c[cndx] = temp + cbeta * c[cndx];
            cndx += ldc;
          }
          cstrt++; /* set index for next row of c */
          astrt++;
        }
      }
      break;
    case 3: /* both matrices tranposed. Combination of cases 1 and 2 */
      /* Transpose matrix b */

      indx_strt = 0;
      bstrt = 0;
      for (j = 0; j < rowsb; j++) {
        indx = indx_strt;
        bndx = bstrt;
        for (i = 0; i < colsb; i++) {
          bufferb[indx] = calpha * b[bndx++];
          indx += rowsb;
        }
        indx_strt++;
        bstrt += ldb;
      }

      /* All of b is now transposed */
      andx = 0;
      cstrt = 0;
      bndx = 0;
      if (cbeta == (double)0.0) {
        for (i = 0; i < colsb; i++) {
          /* Matrix a need not be transposed */
          cndx = cstrt;
          for (j = 0; j < rowsa; j++) {
            temp = 0.0;
            for (k = 0; k < rowsb; k++)
              temp += a[andx + k] * bufferb[bndx + k];
            c[cndx] = temp;
            cndx++;
            andx += lda;
          }
          bndx += rowsb; /* index for next transposed column of b */
          andx = 0;      /* set index for next column of a */
          cstrt += ldc;  /* set index for next row of c */
        }
      } else {
        for (i = 0; i < colsb; i++) {
          /* Matrix a need not be transposed */
          cndx = cstrt;
          for (j = 0; j < rowsa; j++) {
            temp = 0.0;
            for (k = 0; k < rowsb; k++)
              temp += a[andx + k] * bufferb[bndx + k];
            c[cndx] = temp + cbeta * c[cndx];
            cndx++;
            andx += lda;
          }
          bndx += rowsb; /* index for next transposed column of b */
          andx = 0;      /* set index for next column of a */
          cstrt += ldc;  /* set index for next row of c */
        }
      }
    }
  } else {
    switch (tindex) {
    case 0:
      ftn_mnaxnb_real8_(&mra, &ncb, &kab, alpha, a, &lda, b, &ldb, beta, c,
                          &ldc);
      break;
    case 1:
      ftn_mtaxnb_real8_(&mra, &ncb, &kab, alpha, a, &lda, b, &ldb, beta, c,
                          &ldc);
      break;
    case 2:
      ftn_mnaxtb_real8_(&mra, &ncb, &kab, alpha, a, &lda, b, &ldb, beta, c,
                          &ldc);
      break;
    case 3:
      ftn_mtaxtb_real8_(&mra, &ncb, &kab, alpha, a, &lda, b, &ldb, beta, c,
                          &ldc);
    }
  }

}

