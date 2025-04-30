/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "complex.h"
#include "mthdecls.h"

#define SMALL_ROWSA 10
#define SMALL_ROWSB 10
#define SMALL_COLSB 10

/* C prototypes for external Fortran functions */
void ftn_mvmul_cmplx16_(int *, int *, __POINT_T *, __POINT_T *,
                        DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *,
                        __POINT_T *, DOUBLE_COMPLEX_TYPE *,
                        DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *);
void ftn_vmmul_cmplx16_(int *, int *, __POINT_T *, __POINT_T *,
                        DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *,
                        DOUBLE_COMPLEX_TYPE *, __POINT_T *,
                        DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *);
void ftn_mnaxnb_cmplx16_(__POINT_T *, __POINT_T *, __POINT_T *,
                         DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *,
                         __POINT_T *, DOUBLE_COMPLEX_TYPE *, __POINT_T *,
                         DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *,
                         __POINT_T *);
void ftn_mnaxtb_cmplx16_(__POINT_T *, __POINT_T *, __POINT_T *,
                         DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *,
                         __POINT_T *, DOUBLE_COMPLEX_TYPE *, __POINT_T *,
                         DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *,
                         __POINT_T *);
void ftn_mtaxnb_cmplx16_(__POINT_T *, __POINT_T *, __POINT_T *,
                         DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *,
                         __POINT_T *, DOUBLE_COMPLEX_TYPE *, __POINT_T *,
                         DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *,
                         __POINT_T *);
void ftn_mtaxtb_cmplx16_(__POINT_T *, __POINT_T *, __POINT_T *,
                         DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *,
                         __POINT_T *, DOUBLE_COMPLEX_TYPE *, __POINT_T *,
                         DOUBLE_COMPLEX_TYPE *, DOUBLE_COMPLEX_TYPE *,
                         __POINT_T *);

void ENTF90(MMUL_CMPLX16,
            mmul_cmplx16)(int ta, int tb, __POINT_T mra, __POINT_T ncb,
                          __POINT_T kab, DOUBLE_COMPLEX_TYPE *alpha,
                          DOUBLE_COMPLEX_TYPE a[], __POINT_T lda, DOUBLE_COMPLEX_TYPE b[],
                          __POINT_T ldb, DOUBLE_COMPLEX_TYPE *beta,
                          DOUBLE_COMPLEX_TYPE c[], __POINT_T ldc)
{
  /*
   *   Notes on parameters
   *   ta, tb = 0 -> no transpose of matrix
   *   ta, tb = 1 -> transpose of matrix
   *   ta, tb = 2 -> conjugate of matrix
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
  DOUBLE_COMPLEX_TYPE buffera[SMALL_ROWSA * SMALL_ROWSB];
  DOUBLE_COMPLEX_TYPE bufferb[SMALL_COLSB * SMALL_ROWSB];
  DOUBLE_COMPLEX_TYPE temp;
  DOUBLE_COMPLEX_TYPE calpha, cbeta;
  /*
   * Small matrix multiply variables
   */
  int i, ja, j, k;
  int astrt, bstrt, cstrt, andx, bndx, cndx, indx, indx_strt;
  /*
   * We will structure this code a bit different from the real code
   * since there are 9 cases rather than 4.
   * We will switch on ta and then on tb.
   */
  calpha = *alpha;
  cbeta = *beta;
  rowsa = mra;
  colsa = kab;
  rowsb = kab;
  colsb = ncb;
  if (DOUBLE_COMPLEX_EQ_CC(calpha, DOUBLE_COMPLEX_CREATE(0.0, 0.0))) {
    if (DOUBLE_COMPLEX_EQ_CC(cbeta, DOUBLE_COMPLEX_CREATE(0.0, 0.0))) {
      cndx = 0;
      indx_strt = ldc;
      for (j = 0; j < ncb; j++) {
        for (i = 0; i < mra; i++)
          c[cndx + i] = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
        cndx = indx_strt;
        indx_strt += ldc;
      }
    } else {
      cndx = 0;
      indx_strt = ldc;
      for (j = 0; j < ncb; j++) {
        for (i = 0; i < mra; i++)
          c[cndx + i] = DOUBLE_COMPLEX_MUL_CC(cbeta, c[cndx + i]);
        cndx = indx_strt;
        indx_strt += ldc;
      }
    }
    return;
  }

  /*  if( ( tb != 1 ) && ( ncb == 1 ) && ( ldc == 1 ) ){ */
  if ((tb != 1) && (ncb == 1)) {
    /* matrix vector multiply */
    ftn_mvmul_cmplx16_(&ta, &tb, &mra, &kab, alpha, a, &lda, b, beta, c);
    return;
  }
  if ((ta != 1) && (mra == 1)) {
    /* vector matrix multiply */
    ftn_vmmul_cmplx16_(&ta, &tb, &ncb, &kab, alpha, a, b, &ldb, beta, c);
    return;
  }

  // Check for really small matrix sizes
  if ((colsb <= SMALL_COLSB) && (rowsa <= SMALL_ROWSA) &&
      (rowsb <= SMALL_ROWSB)) {
    if (ta == 0) { /* a is normally oriented */
      if (tb == 0) {
        astrt = 0;
        cstrt = 0;
        for (i = 0; i < rowsa; i++) {
          /* Transpose the a row of the a matrix */
          bstrt = 0;
          andx = astrt;
          indx = 0;
          for (ja = 0; ja < colsa; ja++) {
            buffera[indx++] = DOUBLE_COMPLEX_MUL_CC(calpha, a[andx]);
            andx += lda;
          }
          astrt++;
          cndx = cstrt;
          /* Now use the transposed row on all of b */
          if (DOUBLE_COMPLEX_EQ_CC(cbeta, DOUBLE_COMPLEX_CREATE(0.0, 0.0))) {
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              bndx = bstrt;
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(buffera[k], b[bndx++]));
              bstrt += ldb;
              c[cndx] = temp;
              cndx += ldc;
            }
          } else {
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              bndx = bstrt;
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(buffera[k], b[bndx++]));
              bstrt += ldb;
              c[cndx] = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(cbeta, c[cndx]));
              cndx += ldc;
            }
          }
          cstrt++; /* set index for next row of c */
        }
      } else {
        if (tb == 1) { /* b is transposed */
          indx_strt = 0;
          bstrt = 0;
          for (j = 0; j < rowsb; j++) {
            indx = indx_strt;
            bndx = bstrt;
            for (i = 0; i < colsb; i++) {
              bufferb[indx] = b[bndx++];
              //	      	      printf( "( %f, %f )\n", crealf(
              // bufferb[indx] ), cimagf( bufferb[indx] ) );

              indx += rowsb;
            }
            indx_strt++;
            bstrt += ldb;
          }
        } else { /* b is conjugated */
          indx_strt = 0;
          bstrt = 0;
          for (j = 0; j < rowsb; j++) {
            indx = indx_strt;
            bndx = bstrt;
            for (i = 0; i < colsb; i++) {
              bufferb[indx] = conj(b[bndx++]);
              //	      printf( "( %f, %f )\n", crealf( bufferb[indx] ),
              // cimagf( bufferb[indx] ) );
              indx += rowsb;
            }
            indx_strt++;
            bstrt += ldb;
          }
        }

        /* Now muliply the transposed b matrix by a */

        if (DOUBLE_COMPLEX_EQ_CC(cbeta, DOUBLE_COMPLEX_CREATE(0.0, 0.0))) { /* beta == 0.0 */
          astrt = 0;
          indx = 0;
          cstrt = 0;
          for (i = 0; i < rowsa; i++) {
            /* Transpose the a row of the a matrix */
            andx = astrt;
            indx = 0; /* indx will be used for accessing both buffera and
                         bufferb */
            for (ja = 0; ja < colsa; ja++) {
              buffera[indx++] = a[andx];
              andx += lda;
            }
            astrt++;
            cndx = cstrt;
            indx = 0;
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(buffera[k], bufferb[indx++]));
              c[cndx] = DOUBLE_COMPLEX_MUL_CC(calpha, temp);
              cndx += ldc;
              //	      printf( "( %f, %f )\n", crealf( c[cndx] ), cimagf(
              // c[cndx] ) );
            }
            cstrt++; /* set index for next row of c */
            indx_strt = 0;
          }
        } else { /* beta != 0.0 */
          astrt = 0;
          indx = 0;
          cstrt = 0;
          for (i = 0; i < rowsa; i++) {
            /* Transpose the a row of the a matrix */
            andx = astrt;
            indx = 0; /* indx will be used for accessing both buffera and
                         bufferb */
            for (ja = 0; ja < colsa; ja++) {
              buffera[indx++] = a[andx];
              andx += lda;
            }
            astrt++;
            cndx = cstrt;
            indx = 0;
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(buffera[k], bufferb[indx++]));
              c[cndx] = DOUBLE_COMPLEX_ADD_CC(DOUBLE_COMPLEX_MUL_CC(cbeta, c[cndx]), DOUBLE_COMPLEX_MUL_CC(calpha, temp));
              cndx += ldc;
            }
            cstrt++; /* set index for next row of c */
            indx_strt = 0;
          }
        }
      }
    }

    else if (ta == 1) { /* a is transposed */
      if (tb == 0) {
        astrt = 0;
        cstrt = 0;
        if (DOUBLE_COMPLEX_EQ_CC(cbeta, DOUBLE_COMPLEX_CREATE(0.0, 0.0))) { /* beta == 0 */
          for (i = 0; i < rowsa; i++) {
            cndx = cstrt;
            bstrt = 0;
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              bndx = bstrt;
              andx = astrt;
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(a[andx++], b[bndx++]));
              c[cndx] = DOUBLE_COMPLEX_MUL_CC(calpha, temp);
              //	      printf( "( %f, %f )\n", crealf( c[cndx] ), cimagf(
              // c[cndx] ) );

              bstrt += ldb;
              cndx += ldc;
            }
            cstrt++;
            astrt += lda;
            cstrt++; /* set index for next row of c */
          }
        } else { /* beta != 0 */
          astrt = 0;
          cstrt = 0;
          for (i = 0; i < rowsa; i++) {
            cndx = cstrt;
            bstrt = 0;
            ;
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              bndx = bstrt;
              andx = astrt;
              for (k = 0; k < rowsb; k++) {
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(a[andx], b[bndx]));
                andx++;
                bndx++;
              }
              c[cndx] = DOUBLE_COMPLEX_ADD_CC(DOUBLE_COMPLEX_MUL_CC(cbeta, c[cndx]), DOUBLE_COMPLEX_MUL_CC(calpha, temp));
              // printf( "( %f, %f )\n", crealf( c[cndx] ), cimagf( c[cndx] ) );
              bstrt += ldb;
              cndx += ldc;
            }
            cstrt++;
            astrt += lda;
          }
        }
      } else {
        if (tb == 1) { /* b is transposed */
          indx_strt = 0;
          bstrt = 0;
          for (j = 0; j < rowsb; j++) {
            indx = indx_strt;
            bndx = bstrt;
            for (i = 0; i < colsb; i++) {
              bufferb[indx] = DOUBLE_COMPLEX_MUL_CC(calpha, b[bndx++]);
              // printf( "( %f, %f )\n", crealf( bufferb[indx] ), cimagf(
              // bufferb[indx] ) );
              indx += rowsb;
            }
            indx_strt++;
            bstrt += ldb;
          }
        } else { /* b is conjugated */
          indx_strt = 0;
          bstrt = 0;
          for (j = 0; j < rowsb; j++) {
            indx = indx_strt;
            bndx = bstrt;
            for (i = 0; i < colsb; i++) {
              bufferb[indx] = DOUBLE_COMPLEX_MUL_CC(calpha, conj(b[bndx++]));
              //	      printf( "( %f, %f )\n", crealf( bufferb[indx] ),
              // cimagf( bufferb[indx] ) );
              indx += rowsb;
            }
            indx_strt++;
            bstrt += ldb;
          }
        }

        /* Now muliply the transposed b matrix by a, which is transposed */

        if (DOUBLE_COMPLEX_EQ_CC(cbeta, DOUBLE_COMPLEX_CREATE(0.0, 0.0))) { /* beta == 0.0 */
          astrt = 0;
          indx = 0;
          cstrt = 0;
          for (i = 0; i < rowsa; i++) {
            andx = astrt;
            indx = 0; /* indx will be used for accessing both buffera and
                         bufferb */
            cndx = cstrt;
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              andx = astrt;
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(a[andx++], bufferb[indx++]));
              c[cndx] = temp;
              cndx += ldc;
              //	      printf( "( %f, %f )\n", crealf( c[cndx] ), cimagf(
              // c[cndx] ) );
            }
            cstrt++; /* set index for next row of c */
            astrt += lda;
          }
        }

        else { /* beta != 0.0 */
          astrt = 0;
          indx = 0;
          cstrt = 0;
          for (i = 0; i < rowsa; i++) {
            andx = astrt;
            indx = 0; /* indx will be used for accessing both buffera and
                         bufferb */

            cndx = cstrt;
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              andx = astrt;
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(a[andx++], bufferb[indx++]));
              c[cndx] = DOUBLE_COMPLEX_ADD_CC(DOUBLE_COMPLEX_MUL_CC(cbeta, c[cndx]), temp);
              cndx += ldc;
              //	      printf( "( %f, %f )\n", crealf( c[cndx] ), cimagf(
              // c[cndx] ) );
            }
            cstrt++; /* set index for next row of c */
            astrt += lda;
          }
        }
      }
    } else { /* a is conjugated */
      if (tb == 0) {
        astrt = 0;
        cstrt = 0;
        for (i = 0; i < rowsa; i++) {
          /* Transpose the a row of the a matrix */
          bstrt = 0;
          andx = astrt;
          indx = 0;
          for (ja = 0; ja < colsa; ja++) {
            buffera[indx++] = DOUBLE_COMPLEX_MUL_CC(calpha, a[andx]);
            andx += lda;
          }
          astrt++;
          cndx = cstrt;
          /* Now use the transposed row on all of b */
          if (DOUBLE_COMPLEX_EQ_CC(cbeta, DOUBLE_COMPLEX_CREATE(0.0, 0.0))) {
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              bndx = bstrt;
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(buffera[k], b[bndx++]));
              bstrt += ldb;
              c[cndx] = temp;
              cndx += ldc;
            }
            cstrt++; /* set index for next row of c */
          } else {
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              bndx = bstrt;
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(buffera[k], b[bndx++]));
              bstrt += ldb;
              c[cndx] = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(cbeta, c[cndx]));
              cndx += ldc;
            }
            cstrt++; /* set index for next row of c */
          }
        }
      } else {
        if (tb == 1) { /* b is transposed */
          indx_strt = 0;
          bstrt = 0;
          for (j = 0; j < rowsb; j++) {
            indx = indx_strt;
            bndx = bstrt;
            for (i = 0; i < colsb; i++) {
              bufferb[indx] = DOUBLE_COMPLEX_MUL_CC(calpha, b[bndx++]);
              //	      	      printf( "( %f, %f )\n", crealf(
              // bufferb[indx] ), cimagf( bufferb[indx] ) );

              indx += rowsb;
            }
            indx_strt++;
            bstrt += ldb;
          }
        } else { /* b is conjugated */
          indx_strt = 0;
          bstrt = 0;
          for (j = 0; j < rowsb; j++) {
            indx = indx_strt;
            bndx = bstrt;
            for (i = 0; i < colsb; i++) {
              bufferb[indx] = DOUBLE_COMPLEX_MUL_CC(calpha, conj(b[bndx++]));
              //	      printf( "( %f, %f )\n", crealf( bufferb[indx] ),
              // cimagf( bufferb[indx] ) );
              indx += rowsb;
            }
            indx_strt++;
            bstrt += ldb;
          }
        }

        /* Now muliply the transposed b matrix by a */

        if (DOUBLE_COMPLEX_EQ_CC(cbeta, DOUBLE_COMPLEX_CREATE(0.0, 0.0))) { /* beta == 0.0 */
          astrt = 0;
          indx = 0;
          cstrt = 0;
          for (i = 0; i < rowsa; i++) {
            /* Transpose the a row of the a matrix */
            andx = astrt;
            indx = 0; /* indx will be used for accessing both buffera and
                         bufferb */
            for (ja = 0; ja < colsa; ja++) {
              buffera[indx++] = DOUBLE_COMPLEX_MUL_CC(calpha, a[andx]);
              andx += lda;
            }
            astrt++;
            cndx = cstrt;
            indx = 0;
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(buffera[k], bufferb[indx++]));
              c[cndx] = temp;
              cndx += ldc;
              //	      printf( "( %f, %f )\n", crealf( c[cndx] ), cimagf(
              // c[cndx] ) );
            }
            cstrt++; /* set index for next row of c */
            indx_strt = 0;
          }
        } else { /* beta != 0.0 */
          astrt = 0;
          indx = 0;
          cstrt = 0;
          for (i = 0; i < rowsa; i++) {
            /* Transpose the a row of the a matrix */
            andx = astrt;
            indx = 0; /* indx will be used for accessing both buffera and
                         bufferb */
            for (ja = 0; ja < colsa; ja++) {
              buffera[indx++] = DOUBLE_COMPLEX_MUL_CC(calpha, a[andx]);
              andx += lda;
            }
            astrt++;
            cndx = cstrt;
            indx = 0;
            for (j = 0; j < colsb; j++) {
              temp = DOUBLE_COMPLEX_CREATE(0.0, 0.0);
              for (k = 0; k < rowsb; k++)
                temp = DOUBLE_COMPLEX_ADD_CC(temp, DOUBLE_COMPLEX_MUL_CC(buffera[k], bufferb[indx++]));
              c[cndx] = DOUBLE_COMPLEX_ADD_CC(DOUBLE_COMPLEX_MUL_CC(cbeta, c[cndx]), temp);
              //	      printf( "( %f, %f )\n", crealf( c[cndx] ), cimagf(
              // c[cndx] ) );
              cndx += ldc;
            }
            cstrt++; /* set index for next row of c */
            indx_strt = 0;
          }
        }
      }
    }
  }

  else {
    tindex = 3;
    if (ta == 0)
      tindex--;
    if (tb == 0)
      tindex -= 2;
    switch (tindex) {
    case 0:
      ftn_mnaxnb_cmplx16_(&mra, &ncb, &kab, alpha, a, &lda, b, &ldb,
                            beta, c, &ldc);
      break;
    case 1:
      ftn_mtaxnb_cmplx16_(&mra, &ncb, &kab, alpha, a, &lda, b, &ldb,
                            beta, c, &ldc);
      break;
    case 2:
      ftn_mnaxtb_cmplx16_(&mra, &ncb, &kab, alpha, a, &lda, b, &ldb,
                            beta, c, &ldc);
      break;
    case 3:
      ftn_mtaxtb_cmplx16_(&mra, &ncb, &kab, alpha, a, &lda, b, &ldb,
                            beta, c, &ldc);
    }
  }

}
