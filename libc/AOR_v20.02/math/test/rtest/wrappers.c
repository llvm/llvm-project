/*
 * wrappers.c - wrappers to modify output of MPFR/MPC test functions
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "intern.h"

void wrapper_init(wrapperctx *ctx)
{
    int i;
    ctx->nops = ctx->nresults = 0;
    for (i = 0; i < 2; i++) {
        ctx->mpfr_ops[i] = NULL;
        ctx->mpc_ops[i] = NULL;
        ctx->ieee_ops[i] = NULL;
    }
    ctx->mpfr_result = NULL;
    ctx->mpc_result = NULL;
    ctx->ieee_result = NULL;
    ctx->need_regen = 0;
}

void wrapper_op_real(wrapperctx *ctx, const mpfr_t r,
                     int size, const uint32 *ieee)
{
    assert(ctx->nops < 2);
    ctx->mpfr_ops[ctx->nops] = r;
    ctx->ieee_ops[ctx->nops] = ieee;
    ctx->size_ops[ctx->nops] = size;
    ctx->nops++;
}

void wrapper_op_complex(wrapperctx *ctx, const mpc_t c,
                        int size, const uint32 *ieee)
{
    assert(ctx->nops < 2);
    ctx->mpc_ops[ctx->nops] = c;
    ctx->ieee_ops[ctx->nops] = ieee;
    ctx->size_ops[ctx->nops] = size;
    ctx->nops++;
}

void wrapper_result_real(wrapperctx *ctx, mpfr_t r,
                         int size, uint32 *ieee)
{
    assert(ctx->nresults < 1);
    ctx->mpfr_result = r;
    ctx->ieee_result = ieee;
    ctx->size_result = size;
    ctx->nresults++;
}

void wrapper_result_complex(wrapperctx *ctx, mpc_t c,
                            int size, uint32 *ieee)
{
    assert(ctx->nresults < 1);
    ctx->mpc_result = c;
    ctx->ieee_result = ieee;
    ctx->size_result = size;
    ctx->nresults++;
}

int wrapper_run(wrapperctx *ctx, wrapperfunc wrappers[MAXWRAPPERS])
{
    int i;
    for (i = 0; i < MAXWRAPPERS && wrappers[i]; i++)
        wrappers[i](ctx);
    universal_wrapper(ctx);
    return ctx->need_regen;
}

mpfr_srcptr wrapper_get_mpfr(wrapperctx *ctx, int op)
{
    if (op < 0) {
        assert(ctx->mpfr_result);
        return ctx->mpfr_result;
    } else {
        assert(ctx->mpfr_ops[op]);
        return ctx->mpfr_ops[op];
    }
}

const uint32 *wrapper_get_ieee(wrapperctx *ctx, int op)
{
    if (op < 0) {
        assert(ctx->mpfr_result);
        return ctx->ieee_result;
    } else {
        assert(ctx->mpfr_ops[op]);
        return ctx->ieee_ops[op];
    }
}

int wrapper_get_nops(wrapperctx *ctx)
{
    return ctx->nops;
}

int wrapper_get_size(wrapperctx *ctx, int op)
{
    if (op < 0) {
        assert(ctx->mpfr_result || ctx->mpc_result);
        return ctx->size_result;
    } else {
        assert(ctx->mpfr_ops[op] || ctx->mpc_ops[op]);
        return ctx->size_ops[op];
    }
}

int wrapper_is_complex(wrapperctx *ctx, int op)
{
    if (op < 0) {
        assert(ctx->mpfr_result || ctx->mpc_result);
        return ctx->mpc_result != NULL;
    } else {
        assert(ctx->mpfr_ops[op] || ctx->mpc_ops[op]);
        return ctx->mpc_ops[op] != NULL;
    }
}

mpc_srcptr wrapper_get_mpc(wrapperctx *ctx, int op)
{
    if (op < 0) {
        assert(ctx->mpc_result);
        return ctx->mpc_result;
    } else {
        assert(ctx->mpc_ops[op]);
        return ctx->mpc_ops[op];
    }
}

mpfr_srcptr wrapper_get_mpfr_r(wrapperctx *ctx, int op)
{
    if (op < 0) {
        assert(ctx->mpc_result);
        return mpc_realref(ctx->mpc_result);
    } else {
        assert(ctx->mpc_ops[op]);
        return mpc_realref(ctx->mpc_ops[op]);
    }
}

mpfr_srcptr wrapper_get_mpfr_i(wrapperctx *ctx, int op)
{
    if (op < 0) {
        assert(ctx->mpc_result);
        return mpc_imagref(ctx->mpc_result);
    } else {
        assert(ctx->mpc_ops[op]);
        return mpc_imagref(ctx->mpc_ops[op]);
    }
}

const uint32 *wrapper_get_ieee_r(wrapperctx *ctx, int op)
{
    if (op < 0) {
        assert(ctx->mpc_result);
        return ctx->ieee_result;
    } else {
        assert(ctx->mpc_ops[op]);
        return ctx->ieee_ops[op];
    }
}

const uint32 *wrapper_get_ieee_i(wrapperctx *ctx, int op)
{
    if (op < 0) {
        assert(ctx->mpc_result);
        return ctx->ieee_result + 4;
    } else {
        assert(ctx->mpc_ops[op]);
        return ctx->ieee_ops[op] + 2;
    }
}

void wrapper_set_sign(wrapperctx *ctx, uint32 sign)
{
    assert(ctx->mpfr_result);
    ctx->ieee_result[0] |= (sign & 0x80000000U);
}

void wrapper_set_sign_r(wrapperctx *ctx, uint32 sign)
{
    assert(ctx->mpc_result);
    ctx->ieee_result[0] |= (sign & 0x80000000U);
}

void wrapper_set_sign_i(wrapperctx *ctx, uint32 sign)
{
    assert(ctx->mpc_result);
    ctx->ieee_result[4] |= (sign & 0x80000000U);
}

void wrapper_set_nan(wrapperctx *ctx)
{
    assert(ctx->mpfr_result);
    mpfr_set_nan(ctx->mpfr_result);
    ctx->need_regen = 1;
}

void wrapper_set_nan_r(wrapperctx *ctx)
{
    assert(ctx->mpc_result);
    mpfr_set_nan(mpc_realref(ctx->mpc_result)); /* FIXME: better way? */
    ctx->need_regen = 1;
}

void wrapper_set_nan_i(wrapperctx *ctx)
{
    assert(ctx->mpc_result);
    mpfr_set_nan(mpc_imagref(ctx->mpc_result)); /* FIXME: better way? */
    ctx->need_regen = 1;
}

void wrapper_set_int(wrapperctx *ctx, int val)
{
    assert(ctx->mpfr_result);
    mpfr_set_si(ctx->mpfr_result, val, GMP_RNDN);
    ctx->need_regen = 1;
}

void wrapper_set_int_r(wrapperctx *ctx, int val)
{
    assert(ctx->mpc_result);
    mpfr_set_si(mpc_realref(ctx->mpc_result), val, GMP_RNDN);
    ctx->need_regen = 1;
}

void wrapper_set_int_i(wrapperctx *ctx, int val)
{
    assert(ctx->mpc_result);
    mpfr_set_si(mpc_realref(ctx->mpc_result), val, GMP_RNDN);
    ctx->need_regen = 1;
}

void wrapper_set_mpfr(wrapperctx *ctx, const mpfr_t val)
{
    assert(ctx->mpfr_result);
    mpfr_set(ctx->mpfr_result, val, GMP_RNDN);
    ctx->need_regen = 1;
}

void wrapper_set_mpfr_r(wrapperctx *ctx, const mpfr_t val)
{
    assert(ctx->mpc_result);
    mpfr_set(mpc_realref(ctx->mpc_result), val, GMP_RNDN);
    ctx->need_regen = 1;
}

void wrapper_set_mpfr_i(wrapperctx *ctx, const mpfr_t val)
{
    assert(ctx->mpc_result);
    mpfr_set(mpc_realref(ctx->mpc_result), val, GMP_RNDN);
    ctx->need_regen = 1;
}
