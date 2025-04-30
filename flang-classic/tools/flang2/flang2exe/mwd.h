/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef MWD_H_
#define MWD_H_

/** \file
 * \brief Header for mw's dump routines
 */

#include "gbldefs.h"
#include <stdio.h>
#include "symtab.h"

#if DEBUG

/**
   \brief ...
 */
char *getprintnme(int n);

/**
   \brief ...
 */
char *printname(int sptr);

/**
   \brief ...
 */
int putdty(TY_KIND dty);

/**
   \brief ...
 */
void checkfgraph(const char *s);

/**
   \brief ...
 */
void checktags(const char *phase);

/**
   \brief ...
 */
void dbihonly(void);

/**
   \brief ...
 */
void dbih(void);

/**
   \brief ...
 */
void db(int block);

/**
   \brief ...
 */
void ddtype(DTYPE dtype);

/**
   \brief ...
 */
void ddtypes(void);

/**
   \brief ...
 */
void dfih(int f);

/**
   \brief ...
 */
void dflg(void);

/**
   \brief ...
 */
void dgbl(void);

/**
   \brief ...
 */
void dili(int ilix);

/**
   \brief ...
 */
void dilitre(int ilix);

/**
   \brief ...
 */
void dilt(int ilt);

/**
   \brief ...
 */
void dsa(void);

/**
   \brief ...
 */
void ds(int sptr);

/**
   \brief ...
 */
void dss(int l, int u);

/**
   \brief ...
 */
void dsym(int sptr);

/**
   \brief ...
 */
void dsyms(int l, int u);

/**
   \brief ...
 */
void dumpabnd(int abx);

/**
   \brief ...
 */
void dumpabnds(void);

/**
   \brief ...
 */
void dumpacacheinfo(int cx);

/**
   \brief ...
 */
void dumpacexprs(void);

/**
   \brief ...
 */
void dumpacivs(void);

/**
   \brief ...
 */
void dumpacsymt(int first, int last);

/**
   \brief ...
 */
void dumpadef(int def);

/**
   \brief ...
 */
void _dumpadef(int def, int dumpuses);

/**
   \brief ...
 */
void dumpallloops(void);

/**
   \brief ...
 */
void dumpallploops(void);

/**
   \brief ...
 */
void _dumpaloop(int lpx, bool dumpprivate);

/**
   \brief ...
 */
void dumpalooponly(int lpx);

/**
   \brief ...
 */
void dumpanmelist(int nmelist);

/**
   \brief ...
 */
void dumpanmetree(int nmex);

/**
   \brief ...
 */
void dumparef(int arefx);

/**
   \brief ...
 */
void dumparefs(void);

/**
   \brief ...
 */
void dumparefxlist(int arefxlist);

/**
   \brief ...
 */
void dumpause(int use);

/**
   \brief ...
 */
void _dumpause(int use, int dumpdefs);

/**
   \brief ...
 */
void dumpauselist(int use);

/**
   \brief ...
 */
void dumpauses(void);

/**
   \brief ...
 */
void dumpblock(int block);

/**
   \brief ...
 */
void dumpblocks(const char *title);

/**
   \brief ...
 */
void dumpblocksonly(void);

/**
   \brief ...
 */
void _dumpchildaloopsclist(int lpx);

/**
   \brief ...
 */
void _dumpchildaloops(int lpx);

/**
   \brief ...
 */
void dumpddiff(int v);

/**
   \brief ...
 */
void dumpdef(int def);

/**
   \brief ...
 */
void _dumpdef(int def, int dumpuses);

/**
   \brief ...
 */
void dumpdeflist(int deflist);

/**
   \brief ...
 */
void dumpdefnmes(void);

/**
   \brief ...
 */
void dumpdefs(void);

/**
   \brief ...
 */
void dumpdfs(void);

/**
   \brief ...
 */
void dumpdiff(void);

/**
   \brief ...
 */
void dumpdtype(DTYPE dtype);

/**
   \brief ...
 */
void dumpdtypes(void);

/**
   \brief ...
 */
void dumpdvl(int d);

/**
   \brief ...
 */
void dumpdvls(void);

/**
   \brief ...
 */
void dumpfgraph(void);

/**
   \brief ...
 */
void dumpfgraph2file(void);

/**
   \brief ...
 */
void dumpfg(void);

/**
   \brief ...
 */
void dumpfile(int f);

/**
   \brief ...
 */
void dumpfiles(void);

/**
   \brief ...
 */
void dumpfnodehead(int v);

/**
   \brief ...
 */
void dumpfnode(int v);

/**
   \brief ...
 */
void dumpiltdeflist(int iltx);

/**
   \brief ...
 */
void dumpilt(int ilt);

/**
   \brief ...
 */
void dumpiltmrlist(int iltx);

/**
   \brief ...
 */
void dumpilts(void);

/**
   \brief ...
 */
void dumpiltuselist(int iltx);

/**
   \brief ...
 */
void dumpind(int i);

/**
   \brief ...
 */
void dumpinds(void);

/**
   \brief ...
 */
void dumpiv(int ivx);

/**
   \brief ...
 */
void dumpivlist(void);

/**
   \brief ...
 */
void dumpliveinuses(void);

/**
   \brief ...
 */
void dumpliveoutdefs(void);

/**
   \brief ...
 */
void dumplong(void);

/**
   \brief ...
 */
void dumploop(int l);

/**
   \brief ...
 */
void _dumploop(int l, bool dumpldst, bool dumpbits, int nest);

/**
   \brief ...
 */
void dumploopsbv(int bvlen);

/**
   \brief ...
 */
void dumploops(void);

/**
   \brief ...
 */
void _dumplooptree(int l, int nest);

/**
   \brief ...
 */
void dumplooptree(void);

/**
   \brief ...
 */
void dumpmemref(int m);

/**
   \brief ...
 */
void dumpmemrefs(void);

/**
   \brief ...
 */
void _dumpmode2(int mode2);

/**
   \brief ...
 */
void _dumpmode3(int mode3);

/**
   \brief ...
 */
void _dumpmode(int mode);

/**
   \brief ...
 */
void dumpmr(int m);

/**
   \brief ...
 */
void dumpnatural(void);

/**
   \brief ...
 */
void dumpnewdtypes(int olddtavail);

/**
   \brief ...
 */
void dumpnme(int n);

/**
   \brief ...
 */
void _dumpnme(int n, bool dumpdefsuses);

/**
   \brief ...
 */
void dumpnmes(void);

/**
   \brief ...
 */
void dumpnmetree(int n);

/**
   \brief ...
 */
void dumpnnme(int n);

/**
   \brief ...
 */
void dumpnst2(int n);

/**
   \brief ...
 */
void dumpnst(int n);

/**
   \brief ...
 */
void _dumpparentaloopsclist(int lpx);

/**
   \brief ...
 */
void _dumpparentaloops(int lpx);

/**
   \brief ...
 */
void dumpploop(int plpx);

/**
   \brief ...
 */
void _dumpplooptree(int plpx);

/**
   \brief ...
 */
void dumpprivatelist(void);

/**
   \brief ...
 */
void dumpreddeflist(int deflist);

/**
   \brief ...
 */
void dumpred(int r);

/**
   \brief ...
 */
void dumpredlist(int r);

/**
   \brief ...
 */
void dumpreducdeflist(int defx);

/**
   \brief ...
 */
void dumpreducdefs(void);

/**
   \brief ...
 */
void dumpregion(int r);

/**
   \brief ...
 */
void dumpregionnest(int r);

/**
   \brief ...
 */
void dumpregions(void);

/**
   \brief ...
 */
void dumprnest(void);

/**
   \brief ...
 */
void dumpscalar(int s);

/**
   \brief ...
 */
void dumpscalarlist(int ss);

/**
   \brief ...
 */
void dumpscalars(void);

/**
   \brief ...
 */
void dumpshort(void);

/**
   \brief ...
 */
void dumpsizes(void);

/**
   \brief ...
 */
void dumpstmt(int s);

/**
   \brief ...
 */
void dumpstmts(void);

/**
   \brief ...
 */
void dumpstore(int s);

/**
   \brief ...
 */
void dumpstorelist(int s);

/**
   \brief ...
 */
void dumpsub(int s);

/**
   \brief ...
 */
void dumpsubs(void);

/**
   \brief ...
 */
void dumptblock(const char *title, int block);

/**
   \brief ...
 */
void dumpuseddefs(void);

/**
   \brief ...
 */
void dumpuseduses(void);

/**
   \brief ...
 */
void dumpuse(int use);

/**
   \brief ...
 */
void _dumpuse(int use, int dumpdefs);

/**
   \brief ...
 */
void dumpuses(void);

/**
   \brief ...
 */
void dumpvdef(int def);

/**
   \brief ...
 */
void _dumpvdef(int def, int dumpuses);

/**
   \brief ...
 */
void dumpvilt(int iltx);

/**
   \brief ...
 */
void dumpvind(int i);

/**
   \brief ...
 */
void dumpvindl(int i1, int n);

/**
   \brief ...
 */
void dumpvinds(void);

/**
   \brief ...
 */
void dumpvloop(int l);

/**
   \brief ...
 */
void dumpvlooplist(int list);

/**
   \brief ...
 */
void dumpvloops(void);

/**
   \brief ...
 */
void dumpvuse(int use);

/**
   \brief ...
 */
void _dumpvuse(int use, int dumpdefs);

/**
   \brief ...
 */
void pprintnme(int n);

/**
   \brief ...
 */
void printblock(int block);

/**
   \brief ...
 */
void printblockline(int block);

/**
   \brief ...
 */
void printblocksline(void);

/**
   \brief ...
 */
void printblocks(void);

/**
   \brief ...
 */
void dumpdpshape(int shapeid);

/**
   \brief ...
 */
void dumpdppolicy(int policyid);

/**
   \brief ...
 */
void dumpdpshapes();

/**
   \brief ...
 */
void dumpdppolicies();

/**
   \brief ...
 */
void printblockt(int firstblock, int lastblock);

/**
   \brief ...
 */
void printblocktt(int firstblock, int lastblock);

/**
   \brief ...
 */
void printfgraph(void);

/**
   \brief ...
 */
void printfnode(int v);

/**
   \brief ...
 */
void printili(int i);

/**
   \brief ...
 */
void printilt(int i);

/**
   \brief ...
 */
void printloop(int lpx);

/**
   \brief ...
 */
void printnme(int n);

/**
   \brief ...
 */
void printregion(int r);

/**
   \brief ...
 */
void printregionnest(const char *msg, int r);

/**
   \brief ...
 */
void _printregionnest(int r);

/**
   \brief ...
 */
void printregions(void);

/**
   \brief ...
 */
void putarefherelists(int lpx);

/**
   \brief ...
 */
void putareflists(int lpx);

/**
   \brief ...
 */
void putasub(int asubx, int nest);

/**
   \brief ...
 */
void putcoeff(int coefx, int nest);

/**
   \brief ...
 */
void putdtype(DTYPE dtype);

/**
   \brief ...
 */
void _putdtype(DTYPE dtype, int structdepth);

/**
   \brief ...
 */
void putili(const char *name, int ilix);

/**
   \brief ...
 */
void putint1(int d);

/**
   \brief ...
 */
void putlpxareflist(int lpx, const char *listname, int list, int nest);

/**
   \brief ...
 */
void putmode(int mode);

/**
   \brief ...
 */
void putmwline(void);

/**
   \brief ...
 */
void putnestarefherelists(int lpx);

/**
   \brief ...
 */
void _putnestareflists(int lpx, int nest, int flags);

/**
   \brief ...
 */
void putnme(const char *s, int nme);

/**
   \brief ...
 */
void putnnaref(int arefx, int nest);

/**
   \brief ...
 */
void putptelist(int pte);

/**
   \brief ...
 */
void putredtype(int rt);

/**
   \brief ...
 */
void putsclist(const char *name, int list);

/**
   \brief ...
 */
void putsoc(int socptr);

/**
   \brief ...
 */
void _putsubs(const char *name, int s1, int n1);

/**
   \brief ...
 */
void putsubs(int s1, int n1);

/**
   \brief ...
 */
void simpledumpregion(int r);

/**
   \brief ...
 */
void simpledumpstmt(int s);

/**
   \brief ...
 */
void simpleprintregion(int r);

/**
   \brief ...
 */
void simpleprintregionnest(const char *msg, int r);

/**
   \brief ...
 */
void _simpleprintregionnest(int r);

/**
   \brief ...
 */
void stackcheck(void);

/**
   \brief ...
 */
void stackvars(void);

#endif // DEBUG == 1
#endif // MWD_H_
