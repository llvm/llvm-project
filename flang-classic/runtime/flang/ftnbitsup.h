/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


/** \file
 *  \brief extern declarations for ftnbitsup.c
 */

/** \brief
 *  performs circular bit shift
 *  sc > 0 => circular left shift.
 *  sc < 0 => circular right shift.
 */
int
Ftn_ishftc(int val, int sc, int rc);

/** \brief
 *  Ftn_i_iishftc, 16-bit integer
 *  sc > 0 => circular left shift.
 *  sc < 0 => circular right shift.
 */
int
Ftn_i_iishftc(int val, int sc, int rc);

/** 
 *  \brief
 *  performs circular bit shift.
 *  sc > 0 => circular left shift.
 *  sc < 0 => circular right shift.
 */
int Ftn_i_i1shftc(int val, int sc, int rc);

/**
 * \brief
 * moves len bits from pos in src to posd in dest
 */
void Ftn_jmvbits(int src, int pos, int len, int *dest, int posd);

/**
 * \brief
 * moves len bits from pos in src to posd in dest-- dest is 16-bit integer
 */
void Ftn_imvbits(int src, int pos, int len, short int *dest, int posd);

/**
 * \brief
 * zero extends value to 32 bits
 */
int Ftn_jzext(int val, int dt);

/*C* function: Ftn_izext
 * \brief
 * zero extends value to 16 bits
 */
int Ftn_izext(int val, int dt);

