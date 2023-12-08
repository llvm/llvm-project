/*
 * main.c
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>

#include "intern.h"

void gencases(Testable *fn, int number);
void docase(Testable *fn, uint32 *args);
void vet_for_decline(Testable *fn, uint32 *args, uint32 *result, int got_errno_in);
void seed_random(uint32 seed);

int check_declines = 0;
int lib_fo = 0;
int lib_no_arith = 0;
int ntests = 0;

int nargs_(Testable* f) {
    switch((f)->type) {
    case args2:
    case args2f:
    case semi2:
    case semi2f:
    case t_ldexp:
    case t_ldexpf:
    case args1c:
    case args1fc:
    case args1cr:
    case args1fcr:
    case compare:
    case comparef:
        return 2;
    case args2c:
    case args2fc:
        return 4;
    default:
        return 1;
    }
}

static int isdouble(Testable *f)
{
    switch (f->type) {
      case args1:
      case rred:
      case semi1:
      case t_frexp:
      case t_modf:
      case classify:
      case t_ldexp:
      case args2:
      case semi2:
      case args1c:
      case args1cr:
      case compare:
      case args2c:
        return 1;
      case args1f:
      case rredf:
      case semi1f:
      case t_frexpf:
      case t_modff:
      case classifyf:
      case args2f:
      case semi2f:
      case t_ldexpf:
      case comparef:
      case args1fc:
      case args1fcr:
      case args2fc:
        return 0;
      default:
        assert(0 && "Bad function type");
    }
}

Testable *find_function(const char *func)
{
    int i;
    for (i = 0; i < nfunctions; i++) {
        if (func && !strcmp(func, functions[i].name)) {
            return &functions[i];
        }
    }
    return NULL;
}

void get_operand(const char *str, Testable *f, uint32 *word0, uint32 *word1)
{
    struct special {
        unsigned dblword0, dblword1, sglword;
        const char *name;
    } specials[] = {
        {0x00000000,0x00000000,0x00000000,"0"},
        {0x3FF00000,0x00000000,0x3f800000,"1"},
        {0x7FF00000,0x00000000,0x7f800000,"inf"},
        {0x7FF80000,0x00000001,0x7fc00000,"qnan"},
        {0x7FF00000,0x00000001,0x7f800001,"snan"},
        {0x3ff921fb,0x54442d18,0x3fc90fdb,"pi2"},
        {0x400921fb,0x54442d18,0x40490fdb,"pi"},
        {0x3fe921fb,0x54442d18,0x3f490fdb,"pi4"},
        {0x4002d97c,0x7f3321d2,0x4016cbe4,"3pi4"},
    };
    int i;

    for (i = 0; i < (int)(sizeof(specials)/sizeof(*specials)); i++) {
        if (!strcmp(str, specials[i].name) ||
            ((str[0] == '-' || str[0] == '+') &&
             !strcmp(str+1, specials[i].name))) {
            assert(f);
            if (isdouble(f)) {
                *word0 = specials[i].dblword0;
                *word1 = specials[i].dblword1;
            } else {
                *word0 = specials[i].sglword;
                *word1 = 0;
            }
            if (str[0] == '-')
                *word0 |= 0x80000000U;
            return;
        }
    }

    sscanf(str, "%"I32"x.%"I32"x", word0, word1);
}

void dofile(FILE *fp, int translating) {
    char buf[1024], sparebuf[1024], *p;

    /*
     * Command syntax is:
     *
     *  - "seed <integer>" sets a random seed
     *
     *  - "test <function> <ntests>" generates random test lines
     *
     *  - "<function> op1=foo [op2=bar]" generates a specific test
     *  - "func=<function> op1=foo [op2=bar]" does the same
     *  - "func=<function> op1=foo result=bar" will just output the line as-is
     *
     *  - a semicolon or a blank line is ignored
     */
    while (fgets(buf, sizeof(buf), fp)) {
        buf[strcspn(buf, "\r\n")] = '\0';
        strcpy(sparebuf, buf);
        p = buf;
        while (*p && isspace(*p)) p++;
        if (!*p || *p == ';') {
            /* Comment or blank line. Only print if `translating' is set. */
            if (translating)
                printf("%s\n", buf);
            continue;
        }
        if (!strncmp(buf, "seed ", 5)) {
            seed_random(atoi(buf+5));
        } else if (!strncmp(buf, "random=", 7)) {
            /*
             * Copy 'random=on' / 'random=off' lines unconditionally
             * to the output, so that random test failures can be
             * accumulated into a recent-failures-list file and
             * still identified as random-in-origin when re-run the
             * next day.
             */
            printf("%s\n", buf);
        } else if (!strncmp(buf, "test ", 5)) {
            char *p = buf+5;
            char *q;
            int ntests, i;
            q = p;
            while (*p && !isspace(*p)) p++;
            if (*p) *p++ = '\0';
            while (*p && isspace(*p)) p++;
            if (*p)
                ntests = atoi(p);
            else
                ntests = 100;          /* *shrug* */
            for (i = 0; i < nfunctions; i++) {
                if (!strcmp(q, functions[i].name)) {
                    gencases(&functions[i], ntests);
                    break;
                }
            }
            if (i == nfunctions) {
                fprintf(stderr, "unknown test `%s'\n", q);
            }
        } else {
            /*
             * Parse a specific test line.
             */
            uint32 ops[8], result[8];
            int got_op = 0; /* &1 for got_op1, &4 for got_op3 etc. */
            Testable *f = 0;
            char *q, *r;
            int got_result = 0, got_errno_in = 0;

            for (q = strtok(p, " \t"); q; q = strtok(NULL, " \t")) {
                r = strchr(q, '=');
                if (!r) {
                    f = find_function(q);
                } else {
                    *r++ = '\0';

                    if (!strcmp(q, "func"))
                        f = find_function(r);
                    else if (!strcmp(q, "op1") || !strcmp(q, "op1r")) {
                        get_operand(r, f, &ops[0], &ops[1]);
                        got_op |= 1;
                    } else if (!strcmp(q, "op2") || !strcmp(q, "op1i")) {
                        get_operand(r, f, &ops[2], &ops[3]);
                        got_op |= 2;
                    } else if (!strcmp(q, "op2r")) {
                        get_operand(r, f, &ops[4], &ops[5]);
                        got_op |= 4;
                    } else if (!strcmp(q, "op2i")) {
                        get_operand(r, f, &ops[6], &ops[7]);
                        got_op |= 8;
                    } else if (!strcmp(q, "result") || !strcmp(q, "resultr")) {
                        get_operand(r, f, &result[0], &result[1]);
                        got_result |= 1;
                    } else if (!strcmp(q, "resulti")) {
                        get_operand(r, f, &result[4], &result[5]);
                        got_result |= 2;
                    } else if (!strcmp(q, "res2")) {
                        get_operand(r, f, &result[2], &result[3]);
                        got_result |= 4;
                    } else if (!strcmp(q, "errno_in")) {
                        got_errno_in = 1;
                    }
                }
            }

            /*
             * Test cases already set up by the input are not
             * reprocessed by default, unlike the fplib tests. (This
             * is mostly for historical reasons, because we used to
             * use a very slow and incomplete internal reference
             * implementation; now our ref impl is MPFR/MPC it
             * probably wouldn't be such a bad idea, though we'd still
             * have to make sure all the special cases came out
             * right.) If translating==2 (corresponding to the -T
             * command-line option) then we regenerate everything
             * regardless.
             */
            if (got_result && translating < 2) {
                if (f)
                    vet_for_decline(f, ops, result, got_errno_in);
                puts(sparebuf);
                continue;
            }

            if (f && got_op==(1<<nargs_(f))-1) {
                /*
                 * And do it!
                 */
                docase(f, ops);
            }
        }
    }
}

int main(int argc, char **argv) {
    int errs = 0, opts = 1, files = 0, translating = 0;
    unsigned int seed = 1; /* in case no explicit seed provided */

    seed_random(seed);

    setvbuf(stdout, NULL, _IOLBF, BUFSIZ); /* stops incomplete lines being printed when out of time */

    while (--argc) {
        FILE *fp;
        char *p = *++argv;

        if (opts && *p == '-') {
            if(*(p+1) == 0) { /* single -, read from stdin */
                break;
            } else if (!strcmp(p, "-t")) {
                translating = 1;
            } else if (!strcmp(p, "-T")) {
                translating = 2;
            } else if (!strcmp(p, "-c")) {
                check_declines = 1;
            } else if (!strcmp(p, "--")) {
                opts = 0;
            } else if (!strcmp(p,"--seed") && argc > 1 && 1==sscanf(*(argv+1),"%u",&seed)) {
                seed_random(seed);
                argv++; /* next in argv is seed value, so skip */
                --argc;
            } else if (!strcmp(p, "-fo")) {
                lib_fo = 1;
            } else if (!strcmp(p, "-noarith")) {
                lib_no_arith = 1;
            } else {
                fprintf(stderr,
                        "rtest: ignoring unrecognised option '%s'\n", p);
                errs = 1;
            }
        } else {
            files = 1;
            if (!errs) {
                fp = fopen(p, "r");
                if (fp) {
                    dofile(fp, translating);
                    fclose(fp);
                } else {
                    perror(p);
                    errs = 1;
                }
            }
        }
    }

    /*
     * If no filename arguments, use stdin.
     */
    if (!files && !errs) {
        dofile(stdin, translating);
    }

    if (check_declines) {
        fprintf(stderr, "Tests expected to run: %d\n", ntests);
        fflush(stderr);
    }

    return errs;
}
