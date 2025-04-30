/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*  define structures and data for keyword processing: */

typedef struct {
  const char *keytext; /* keyword text in lower case */
  int toktyp;          /* token id (as used in parse tables) */
  LOGICAL nonstandard; /* TRUE if nonstandard (extension to f90) */
} KWORD;

typedef struct {
  int kcount;  /* number of keywords in this table */
  KWORD *kwds; /* pointer to first in array of KWORD */
               /* the following members are filled in by init_ktable() to record
                * the indices of the first and last keywords beginning with a
                * certain letter.   If first[] is zero, there does not exist a
                * keyword which begins with the respective letter.  A nonzero
                * value is the index into the keyword table.
                */
  short first[26]; /* indexed by ('a' ... 'z') - 'a' */
  short last[26];  /* indexed by ('a' ... 'z') - 'a' */
} KTABLE;

/* NOTE:  When entering keywords in the tables, two or more consecutive tokens
 * that can be seen in the grammar should be combined into a single token.
 * This is because the keyword search routine is not designed to extract two
 * or more tokens from a single identifier. For example, consider the
 * statement  DO WHILE ( <expr> ).  One way of looking at the statement is
 * that it begins with the two keywords DO and WHILE.  The routine alpha
 * sees the name DOWHILE; relevant tokens are "do", "doublecomplex", and
 * "doubleprecision".  When the keyword search routine looks at "dowhile",
 * it determines that it should look AFTER "doubleprecision" --- the character
 * 'w' is greater than 'u'.  Consequently, no keyword, in particular DO, is
 * found.   For the statement, DO 10 WHILE ( <expr> ), the keyword DO is
 * found in "do10while";  it is determined that it must look BEFORE
 * "doublecomplex" ('1' < 'u').  DO statements of the form,
 *     DO <var> =
 * are not a problem because of the '='; the keyword DO is found as a special
 * case in routine alpha since there is an "exposed" equals sign.
 *
 * If a keyword is a prefix of another keyword, it is possible that the
 * keyword search routine will not find the 'shorter' prefix of an identifier.
 * E.g.,  real and realign are keywords
 *        The identifier realpi will not be found since 'realpi' > 'realign';
 *        the search routine searches past realign instead of before.
 * For cases like this, the keyword table must only contain the 'shorter'
 * prefix; it's up to alpha() to determine look for the additional keywords;
 * e.g., 'realign' is 'real' followed by 'ign'.
 *
 */

/*
 * When the input is freeform, certain keywords may contain blanks.
 * For those whose 'first' ident is also a keyword, the processing in
 * alpha() will search ahead to determine if what follows is an ident
 * which can be combined with 'first' to form a keyword; e.g.,
 *     end if, end program, etc.
 * There are a few whose 'first' ident is not a keyword.  For these,
 * define special macros (TKF_ ...) and initialize in the keyword
 * table and let alpha() do the rest.  (NOTE that 0 is returned by
 * keyword() if a name is not found in a keyword table).
 */
#define TKF_ARRAY -1
#define TKF_ATOMIC -2
#define TKF_CANCELLATION -3
#define TKF_DISTPAR -4
#define TKF_DOCONCURRENT -5
#define TKF_DOUBLE -6
#define TKF_DOWHILE -7
#define TKF_ENDDISTPAR -8
#define TKF_GO -9
#define TKF_NO -10
#define TKF_SELECT -11
#define TKF_TARGETENTER -12
#define TKF_TARGETEXIT -13

static KWORD t1[] = {         /* normal keyword table */
                     {"", 0, 0}, /* a keyword index must be nonzero */
                     {"abstract", TK_ABSTRACT, 0},
                     {"accept", TK_ACCEPT, 0},
                     {"align", TK_ALIGN, 0},
                     {"allocatable", TK_ALLOCATABLE, 0},
                     {"allocate", TK_ALLOCATE, 0},
                     {"array", TKF_ARRAY, 0},
                     {"assign", TK_ASSIGN, 0},
                     {"associate", TK_ASSOCIATE, 0},
                     {"asynchronous", TK_ASYNCHRONOUS, 0},
                     {"backspace", TK_BACKSPACE, 0},
                     {"bind", TK_BIND, 0},
                     {"block", TK_BLOCK, 0},
                     {"blockdata", TK_BLOCKDATA, 0},
                     {"byte", TK_BYTE, 0},
                     {"call", TK_CALL, 0},
                     {"case", TK_CASE, 0},
                     {"casedefault", TK_CASEDEFAULT, 0},
                     {"character", TK_CHARACTER, 0},
                     {"class", TK_CLASS, 0},
                     {"close", TK_CLOSE, 0},
                     {"common", TK_COMMON, 0},
                     {"complex", TK_COMPLEX, 0},
                     {"concurrent", TK_CONCURRENT, 0},
                     {"contains", TK_CONTAINS, 0},
                     {"contiguous", TK_CONTIGUOUS, 0},
                     {"continue", TK_CONTINUE, 0},
                     {"cycle", TK_CYCLE, 0},
                     {"data", TK_DATA, 0},
                     {"deallocate", TK_DEALLOCATE, 0},
                     {"decode", TK_DECODE, 0},
                     {"default", TK_DEFAULT, 0},
                     {"dimension", TK_DIMENSION, 0},
                     {"do", TK_DO, 0},
                     {"doconcurrent", TKF_DOCONCURRENT, 0},
                     {"double", TKF_DOUBLE, 0},
                     {"doublecomplex", TK_DBLECMPLX, 0},
                     {"doubleprecision", TK_DBLEPREC, 0},
                     {"dowhile", TKF_DOWHILE, 0},
                     {"elemental", TK_ELEMENTAL, 0},
                     {"else", TK_ELSE, 0},
                     {"elseforall", TK_FORALL, 0},
                     {"elseif", TK_ELSEIF, 0},
                     {"elsewhere", TK_ELSEWHERE, 0},
                     {"encode", TK_ENCODE, 0},
                     {"end", TK_ENDSTMT, 0},
                     {"endassociate", TK_ENDASSOCIATE, 0},
                     {"endblock", TK_ENDBLOCK, 0},
                     {"endblockdata", TK_ENDBLOCKDATA, 0},
                     {"enddo", TK_ENDDO, 0},
                     {"endenum", TK_ENDENUM, 0},
                     {"endfile", TK_ENDFILE, 0},
                     {"endforall", TK_ENDFORALL, 0},
                     {"endfunction", TK_ENDFUNCTION, 0},
                     {"endif", TK_ENDIF, 0},
                     {"endinterface", TK_ENDINTERFACE, 0},
                     {"endmap", TK_ENDMAP, 0},
                     {"endmodule", TK_ENDMODULE, 0},
                     {"endprocedure", TK_ENDPROCEDURE, 0},
                     {"endprogram", TK_ENDPROGRAM, 0},
                     {"endselect", TK_ENDSELECT, 0},
                     {"endstructure", TK_ENDSTRUCTURE, 0},
                     {"endsubmodule", TK_ENDSUBMODULE, 0},
                     {"endsubroutine", TK_ENDSUBROUTINE, 0},
                     {"endtype", TK_ENDTYPE, 0},
                     {"endunion", TK_ENDUNION, 0},
                     {"endwhere", TK_ENDWHERE, 0},
                     {"entry", TK_ENTRY, 0},
                     {"enum", TK_ENUM, 0},
                     {"enumerator", TK_ENUMERATOR, 0},
                     {"equivalence", TK_EQUIV, 0},
                     {"errorstop", TK_ERRORSTOP, 0},
                     {"exit", TK_EXIT, 0},
                     {"extends", TK_EXTENDS, 0},
                     {"external", TK_EXTERNAL, 0},
                     {"final", TK_FINAL, 0},
                     {"flush", TK_FLUSH, 0},
                     {"forall", TK_FORALL, 0},
                     {"format", TK_FORMAT, 0},
                     {"function", TK_FUNCTION, 0},
                     {"generic", TK_GENERIC, 0},
                     {"go", TKF_GO, 0},
                     {"goto", TK_GOTO, 0},
                     {"if", TK_IF, 0},
                     {"implicit", TK_IMPLICIT, 0},
                     {"import", TK_IMPORT, 0},
                     {"impure", TK_IMPURE, 0},
                     {"include", TK_INCLUDE, 0},
                     {"independent", TK_INDEPENDENT, 0},
                     {"inquire", TK_INQUIRE, 0},
                     {"integer", TK_INTEGER, 0},
                     {"intent", TK_INTENT, 0},
                     {"interface", TK_INTERFACE, 0},
                     {"intrinsic", TK_INTRINSIC, 0},
                     {"local", TK_LOCAL, 0},
                     {"local_init", TK_LOCAL_INIT, 0},
                     {"logical", TK_LOGICAL, 0},
                     {"map", TK_MAP, 0},
                     {"module", TK_MODULE, 0},
                     {"namelist", TK_NAMELIST, 0},
                     {"ncharacter", TK_NCHARACTER, 0},
                     {"no", TKF_NO, 0},
                     {"non_intrinsic", TK_NON_INTRINSIC, 0},
                     {"none", TK_NONE, 0},
                     {"nopass", TK_NOPASS, 0},
                     {"nosequence", TK_NOSEQUENCE, 0},
                     {"nullify", TK_NULLIFY, 0},
                     {"open", TK_OPEN, 0},
                     {"optional", TK_OPTIONAL, 0},
                     {"options", TK_OPTIONS, 0},
                     {"parameter", TK_PARAMETER, 0},
                     {"pass", TK_PASS, 0},
                     {"pause", TK_PAUSE, 0},
                     {"pointer", TK_POINTER, 0},
                     {"print", TK_PRINT, 0},
                     {"private", TK_PRIVATE, 0},
                     {"procedure", TK_PROCEDURE, 0},
                     {"program", TK_PROGRAM, 0},
                     {"protected", TK_PROTECTED, 0},
                     {"public", TK_PUBLIC, 0},
                     {"pure", TK_PURE, 0},
                     {"quiet", TK_QUIET, 0},
                     {"read", TK_READ, 0},
                     {"real", TK_REAL, 0},
                     {"record", TK_RECORD, 0},
                     {"recursive", TK_RECURSIVE, 0},
                     {"return", TK_RETURN, 0},
                     {"rewind", TK_REWIND, 0},
                     {"save", TK_SAVE, 0},
                     {"select", TKF_SELECT, 0},
                     {"selectcase", TK_SELECTCASE, 0},
                     {"selecttype", TK_SELECTTYPE, 0},
                     {"sequence", TK_SEQUENCE, 0},
                     {"shared", TK_SHARED, 0},
                     {"stop", TK_STOP, 0},
                     {"structure", TK_STRUCTURE, 0},
                     {"submodule", TK_SUBMODULE, 0},
                     {"subroutine", TK_SUBROUTINE, 0},
                     {"target", TK_TARGET, 0},
                     {"then", TK_THEN, 0},
                     {"type", TK_TYPE, 0},
                     {"union", TK_UNION, 0},
                     {"use", TK_USE, 0},
                     {"value", TK_VALUE, 0},
                     {"volatile", TK_VOLATILE, 0},
                     {"wait", TK_WAIT, 0},
                     {"where", TK_WHERE, 0},
                     {"while", TK_WHILE, 0},
                     {"write", TK_WRITE, 0}};

static KWORD t2[] = {         /* logical keywords */
                     {"", 0, 0}, /* a keyword index must be nonzero */
                     {"a", TK_AND, TRUE},
                     {"and", TK_AND, FALSE},
                     {"eq", TK_EQ, FALSE},
                     {"eqv", TK_EQV, FALSE},
                     {"f", TK_LOGCONST, TRUE},
                     {"false", TK_LOGCONST, FALSE},
                     {"ge", TK_GE, FALSE},
                     {"gt", TK_GT, FALSE},
                     {"le", TK_LE, FALSE},
                     {"lt", TK_LT, FALSE},
                     {"n", TK_NOTX, TRUE},
                     {"ne", TK_NE, FALSE},
                     {"neqv", TK_NEQV, FALSE},
                     {"not", TK_NOT, FALSE},
                     {"o", TK_ORX, TRUE},
                     {"or", TK_OR, FALSE},
                     {"t", TK_LOGCONST, TRUE},
                     {"true", TK_LOGCONST, FALSE},
                     {"x", TK_XORX, TRUE},
                     {"xor", TK_XOR, TRUE}};

static KWORD t3[] = {
    /* I/O keywords and ALLOCATE keywords */
    {"", 0, 0}, /* a keyword index must be nonzero */
    {"access", TK_ACCESS, 0},
    {"action", TK_ACTION, 0},
    {"advance", TK_ADVANCE, 0},
    {"align", TK_ALIGN, 0}, /* ... used in ALLOCATE stmt */
    {"asynchronous", TK_ASYNCHRONOUS, 0},
    {"blank", TK_BLANK, 0},
    {"convert", TK_CONVERT, 0},
    {"decimal", TK_DECIMAL, 0},
    {"delim", TK_DELIM, 0},
    {"direct", TK_DIRECT, 0},
    {"disp", TK_DISPOSE, 0},
    {"dispose", TK_DISPOSE, 0},
    {"encoding", TK_ENCODING, 0},
    {"end", TK_END, 0},
    {"eor", TK_EOR, 0},
    {"err", TK_ERR, 0},
    {"errmsg", TK_ERRMSG, 0}, /* ... used in ALLOCATE stmt */
    {"exist", TK_EXIST, 0},
    {"file", TK_FILE, 0},
    {"fmt", TK_FMT, 0},
    {"form", TK_FORM, 0},
    {"formatted", TK_FMTTD, 0},
    {"id", TK_ID, 0},
    {"iolength", TK_IOLENGTH, 0},
    {"iomsg", TK_IOMSG, 0},
    {"iostat", TK_IOSTAT, 0},
    {"mold", TK_MOLD, 0},
    {"name", TK_NAME, 0},
    {"named", TK_NAMED, 0},
    {"newunit", TK_NEWUNIT, 0},
    {"nextrec", TK_NEXTREC, 0},
    {"nml", TK_NML, 0},
    {"number", TK_NUMBER, 0},
    {"opened", TK_OPENED, 0},
    {"pad", TK_PAD, 0},
    {"pending", TK_PENDING, 0},
    {"pinned", TK_PINNED, 0}, /* ... used in ALLOCATE stmt */
    {"pos", TK_POS, 0},
    {"position", TK_POSITION, 0},
    {"read", TK_READ, 0},
    {"readonly", TK_READONLY, 0},
    {"readwrite", TK_READWRITE, 0},
    {"rec", TK_REC, 0},
    {"recl", TK_RECL, 0},
    {"recordsize", TK_RECL, 0}, /* ... identical to RECL    */
    {"round", TK_ROUND, 0},
    {"sequential", TK_SEQUENTIAL, 0},
    {"shared", TK_SHARED, 0},
    {"sign", TK_SIGN, 0},
    {"size", TK_SIZE, 0},
    {"source", TK_SOURCE, 0}, /* ... used in ALLOCATE stmt */
    {"stat", TK_STAT, 0},     /* ... used in ALLOCATE and DEALLOCATE stmts */
    {"status", TK_STATUS, 0},
    {"stream", TK_STREAM, 0},
    {"type", TK_STATUS, 0}, /* ... identical to STATUS  */
    {"unformatted", TK_UNFORMATTED, 0},
    {"unit", TK_UNIT, 0},
    {"write", TK_WRITE, 0},
};

static KWORD t4[] = {         /* keywords appearing within a FORMAT stmt: */
                     {"", 0, 0}, /* a keyword index must be nonzero */
                     /* {"$", TK_DOLLAR, 0}, special case in alpha() */
                     {"a", TK_A, 0},
                     {"b", TK_B, 0},
                     {"bn", TK_BN, 0},
                     {"bz", TK_BZ, 0},
                     {"d", TK_D, 0},
                     {"dc", TK_DC, 0},
                     {"dp", TK_DP, 0},
                     {"dt", TK_DT, 0},
                     {"e", TK_E, 0},
                     {"en", TK_EN, 0},
                     {"es", TK_ES, 0},
                     {"f", TK_F, 0},
                     {"g", TK_G, 0},
                     {"i", TK_I, 0},
                     {"l", TK_L, 0},
                     {"n", TK_N, 0},
                     {"o", TK_O, 0},
                     {"p", TK_P, 0},
                     {"q", TK_Q, 0},
                     {"s", TK_S, 0},
                     {"rc", TK_RC, 0},
                     {"rd", TK_RD, 0},
                     {"rn", TK_RN, 0},
                     {"rp", TK_RP, 0},
                     {"ru", TK_RU, 0},
                     {"rz", TK_RZ, 0},
                     {"sp", TK_SP, 0},
                     {"ss", TK_SS, 0},
                     {"t", TK_T, 0},
                     {"tl", TK_TL, 0},
                     {"tr", TK_TR, 0},
                     {"x", TK_X, 0},
                     {"z", TK_Z, 0}};

static KWORD t5[] = {
    /* keywords appearing within PARALLEL directives */
    {"", 0, 0}, /* a keyword index must be nonzero */
    {"aligned", TK_ALIGNED, 0},
    {"capture", TK_CAPTURE, 0},
    {"chunk", TK_CHUNK, 0},
    {"collapse", TK_COLLAPSE, 0},
    {"compare", TK_COMPARE, 0},
    {"copyin", TK_COPYIN, 0},
    {"copyprivate", TK_COPYPRIVATE, 0},
    {"default", TK_DEFAULT, 0},
    {"defaultmap", TK_DEFAULTMAP, 0},
    {"depend", TK_DEPEND, 0},
    {"device", TK_DEVICE, 0},
    {"dist_schedule", TK_DIST_SCHEDULE, 0},
    {"final", TK_FINAL, 0},
    {"firstprivate", TK_FIRSTPRIVATE, 0},
    {"from", TK_FROM, 0},
    {"grainsize", TK_GRAINSIZE, 0},
    {"if", TK_IF, 0},
    {"inbranch", TK_INBRANCH, 0},
    {"is_device_ptr", TK_IS_DEVICE_PTR, 0},
    {"lastlocal", TK_LASTPRIVATE, 0},
    {"lastprivate", TK_LASTPRIVATE, 0},
    {"linear", TK_LINEAR, 0},
    {"link", TK_LINK, 0},
    {"local", TK_PRIVATE, 0},
    {"map", TK_MP_MAP, 0},
    {"mergeable", TK_MERGEABLE, 0},
    {"mp_schedtype", TK_MP_SCHEDTYPE, 0},
    {"nogroup", TK_NOGROUP, 0},
    {"notinbranch", TK_NOTINBRANCH, 0},
    {"nowait", TK_NOWAIT, 0},
    {"num_tasks", TK_NUM_TASKS, 0},
    {"num_teams", TK_NUM_TEAMS, 0},
    {"num_threads", TK_NUM_THREADS, 0},
    {"ordered", TK_ORDERED, 0},
    {"priority", TK_PRIORITY, 0},
    {"private", TK_PRIVATE, 0},
    {"proc_bind", TK_PROC_BIND, 0},
    {"read", TK_READ, 0},
    {"reduction", TK_REDUCTION, 0},
    {"safelen", TK_SAFELEN, 0},
    {"schedule", TK_SCHEDULE, 0},
    {"seq_cst", TK_SEQ_CST, 0},
    {"share", TK_SHARED, 0},
    {"shared", TK_SHARED, 0},
    {"simd", TK_SIMD, 0},
    {"simdlen", TK_SIMDLEN, 0},
    {"thread_limit", TK_THREAD_LIMIT, 0},
    {"threads", TK_THREADS, 0},
    {"to", TK_TO, 0},
    {"uniform", TK_UNIFORM, 0},
    {"untied", TK_UNTIED, 0},
    {"update", TK_UPDATE, 0},
    {"write", TK_WRITE, 0},
};

static KWORD t6[] = {
    /* keywords beginning OpenMP/PARALLEL directives */
    {"", 0, 0}, /* a keyword index must be nonzero */
    {"atomic", TK_MP_ATOMIC, 0},
    {"barrier", TK_MP_BARRIER, 0},
    {"cancel", TK_MP_CANCEL, 0},
    {"cancellation", TKF_CANCELLATION, 0},
    {"cancellationpoint", TK_MP_CANCELLATIONPOINT, 0},
    {"critical", TK_MP_CRITICAL, 0},
    {"declare", TK_DECLARE, 0},
    {"declarereduction", TK_MP_DECLAREREDUCTION, 0},
    {"declaresimd", TK_MP_DECLARESIMD, 0},
    {"declaretarget", TK_MP_DECLARETARGET, 0},
    {"distribute", TK_MP_DISTRIBUTE, 0},
    {"distributeparallel", TKF_DISTPAR, 0},
    {"distributeparalleldo", TK_MP_DISTPARDO, 0},
    {"distributeparalleldosimd", TK_MP_DISTPARDOSIMD, 0},
    {"do", TK_MP_PDO, 0},
    {"doacross", TK_MP_DOACROSS, 0},
    {"dosimd", TK_MP_DOSIMD, 0},
    {"end", TK_ENDSTMT, 0},
    {"endatomic", TK_MP_ENDATOMIC, 0},
    {"endcritical", TK_MP_ENDCRITICAL, 0},
    {"enddistribute", TK_MP_ENDDISTRIBUTE, 0},
    {"enddistributeparallel", TKF_ENDDISTPAR, 0},
    {"enddistributeparalleldo", TK_MP_ENDDISTPARDO, 0},
    {"enddistributeparalleldosimd", TK_MP_ENDDISTPARDOSIMD, 0},
    {"enddo", TK_MP_ENDPDO, 0},
    {"enddosimd", TK_MP_ENDDOSIMD, 0},
    {"endmaster", TK_MP_ENDMASTER, 0},
    {"endordered", TK_MP_ENDORDERED, 0},
    {"endparallel", TK_MP_ENDPARALLEL, 0},
    {"endparalleldo", TK_MP_ENDPARDO, 0},
    {"endparalleldosimd", TK_MP_ENDPARDOSIMD, 0},
    {"endparallelsections", TK_MP_ENDPARSECTIONS, 0},
    {"endparallelworkshare", TK_MP_ENDPARWORKSHR, 0},
    {"endsections", TK_MP_ENDSECTIONS, 0},
    {"endsimd", TK_MP_ENDSIMD, 0},
    {"endsingle", TK_MP_ENDSINGLE, 0},
    {"endtarget", TK_MP_ENDTARGET, 0},
    {"endtargetdata", TK_MP_ENDTARGETDATA, 0},
    {"endtargetparallel", TK_MP_ENDTARGPAR, 0},
    {"endtargetparalleldo", TK_MP_ENDTARGPARDO, 0},
    {"endtargetparalleldosimd", TK_MP_ENDTARGPARDOSIMD, 0},
    {"endtargetsimd", TK_MP_ENDTARGSIMD, 0},
    {"endtargetteams", TK_MP_ENDTARGTEAMS, 0},
    {"endtargetteamsdistribute", TK_MP_ENDTARGTEAMSDIST, 0},
    {"endtargetteamsdistributeparalleldo", TK_MP_ENDTARGTEAMSDISTPARDO, 0},
    {"endtargetteamsdistributeparalleldosimd", TK_MP_ENDTARGTEAMSDISTPARDOSIMD, 0},
    {"endtargetteamsdistributesimd", TK_MP_ENDTARGTEAMSDISTSIMD, 0},
    {"endtask", TK_MP_ENDTASK, 0},
    {"endtaskgroup", TK_MP_ENDTASKGROUP, 0},
    {"endtaskloop", TK_MP_ENDTASKLOOP, 0},
    {"endtaskloopsimd", TK_MP_ENDTASKLOOPSIMD, 0},
    {"endteams", TK_MP_ENDTEAMS, 0},
    {"endteamsdistribute", TK_MP_ENDTEAMSDIST, 0},
    {"endteamsdistributeparalleldo", TK_MP_ENDTEAMSDISTPARDO, 0},
    {"endteamsdistributeparalleldosimd", TK_MP_ENDTEAMSDISTPARDOSIMD, 0},
    {"endteamsdistributesimd", TK_MP_ENDTEAMSDISTSIMD, 0},
    {"endworkshare", TK_MP_ENDWORKSHARE, 0},
    {"flush", TK_MP_FLUSH, 0},
    {"master", TK_MP_MASTER, 0},
    {"ordered", TK_MP_ORDERED, 0},
    {"parallel", TK_MP_PARALLEL, 0},
    {"paralleldo", TK_MP_PARDO, 0},
    {"paralleldosimd", TK_MP_PARDOSIMD, 0},
    {"parallelsections", TK_MP_PARSECTIONS, 0},
    {"parallelworkshare", TK_MP_PARWORKSHR, 0},
    {"section", TK_MP_SECTION, 0},
    {"sections", TK_MP_SECTIONS, 0},
    {"simd", TK_MP_SIMD, 0},
    {"single", TK_MP_SINGLE, 0},
    {"target", TK_MP_TARGET, 0},
    {"targetdata", TK_MP_TARGETDATA, 0},
    {"targetenter", TKF_TARGETENTER, 0},
    {"targetenterdata", TK_MP_TARGETENTERDATA, 0},
    {"targetexit", TKF_TARGETEXIT, 0},
    {"targetexitdata", TK_MP_TARGETEXITDATA, 0},
    {"targetparallel", TK_MP_TARGPAR, 0},
    {"targetparalleldo", TK_MP_TARGPARDO, 0},
    {"targetparalleldosimd", TK_MP_TARGPARDOSIMD, 0},
    {"targetsimd", TK_MP_TARGSIMD, 0},
    {"targetteams", TK_MP_TARGTEAMS, 0},
    {"targetteamsdistribute", TK_MP_TARGTEAMSDIST, 0},
    {"targetteamsdistributeparalleldo", TK_MP_TARGTEAMSDISTPARDO, 0},
    {"targetteamsdistributeparalleldosimd", TK_MP_TARGTEAMSDISTPARDOSIMD, 0},
    {"targetteamsdistributesimd", TK_MP_TARGTEAMSDISTSIMD, 0},
    {"targetupdate", TK_MP_TARGETUPDATE, 0},
    {"task", TK_MP_TASK, 0},
    {"taskgroup", TK_MP_TASKGROUP, 0},
    {"taskloop", TK_MP_TASKLOOP, 0},
    {"taskloopsimd", TK_MP_TASKLOOPSIMD, 0},
    {"taskwait", TK_MP_TASKWAIT, 0},
    {"taskyield", TK_MP_TASKYIELD, 0},
    {"teams", TK_MP_TEAMS, 0},
    {"teamsdistribute", TK_MP_TEAMSDIST, 0},
    {"teamsdistributeparalleldo", TK_MP_TEAMSDISTPARDO, 0},
    {"teamsdistributeparalleldosimd", TK_MP_TEAMSDISTPARDOSIMD, 0},
    {"teamsdistributesimd", TK_MP_TEAMSDISTSIMD, 0},
    {"threadprivate", TK_MP_THREADPRIVATE, 0},
    {"workshare", TK_MP_WORKSHARE, 0},
};

static KWORD t7[] = {
    /* keywords which begin a 'cdec$' directive */
    {"", 0, 0}, /* a keyword index must be nonzero */
    {"alias", TK_ALIAS, 0},
    {"attributes", TK_ATTRIBUTES, 0},
    {"craydistributepoint", TK_DISTRIBUTEPOINT, 0},
    {"distribute", TK_DISTRIBUTE, 0},
    {"distributepoint", TK_DISTRIBUTEPOINT, 0},
};

static KWORD t8[] = {
    /* keywords which begin other directives */
    {"", 0, 0}, /* a keyword index must be nonzero */
    {"local", TK_LOCAL, 0},
    {"prefetch", TK_PREFETCH, 0},
};

static KWORD t9[] = {
    /* keywords for parsed PGI pragmas */
    {"", 0, 0}, /* a keyword index must be nonzero */
    {"defaultkind", TK_DFLT, 0},
    {"ignore_tkr", TK_IGNORE_TKR, 0},
    {"movedesc", TK_MOVEDESC, 0},
    {"prefetch", TK_PREFETCH, 0},
};

static KWORD t11[] = {
    /* keywords for kernel directives */
    {"", 0, 0}, /* a keyword index must be nonzero */
    {"do", TK_DO, 0},
    {"kernel", TK_KERNEL, 0},
    {"nowait", TK_NOWAIT, 0},
};

static KWORD t12[] = {
    {"", 0, 0}, /* a keyword index must be nonzero */
    {"compare", TK_PGICOMPARE, 0},
};

/* ****  NOTE -- each of these must appear in a call to init_ktable() in
 *               scan_init().
 */
static KTABLE normalkw = {sizeof(t1) / sizeof(KWORD), &t1[0], {}, {}};
static KTABLE logicalkw = {sizeof(t2) / sizeof(KWORD), &t2[0], {}, {}};
static KTABLE iokw = {sizeof(t3) / sizeof(KWORD), &t3[0], {}, {}};
static KTABLE formatkw = {sizeof(t4) / sizeof(KWORD), &t4[0], {}, {}};
static KTABLE parallelkw = {sizeof(t5) / sizeof(KWORD), &t5[0], {}, {}};
static KTABLE parbegkw = {sizeof(t6) / sizeof(KWORD), &t6[0], {}, {}};
static KTABLE deckw = {sizeof(t7) / sizeof(KWORD), &t7[0], {}, {}};
static KTABLE pragma_kw = {sizeof(t8) / sizeof(KWORD), &t8[0], {}, {}};
static KTABLE ppragma_kw = {sizeof(t9) / sizeof(KWORD), &t9[0], {}, {}};
static KTABLE kernel_kw = {sizeof(t11) / sizeof(KWORD), &t11[0], {}, {}};
static KTABLE pgi_kw = {sizeof(t12) / sizeof(KWORD), &t12[0], {}, {}};

/* char classification macros */

#undef _CS
#undef _DI
#undef _BL
#undef _HD
#undef _HO

#define _CS 1  /* alpha symbol */
#define _DI 2  /* digit */
#define _BL 4  /* blank */
#define _HD 8  /* hex digit */
#define _HO 16 /* Hollerith constant indicator */

#undef iscsym
#define iscsym(c) (ctable[c] & _CS)
#undef isblank
#define isblank(c) (ctable[c] & _BL)
#undef iswhite
#define iswhite(c) ((c) <= ' ')
#define ishex(c) (ctable[c] & (_HD | _DI))
#define isident(c) (ctable[c] & (_CS | _DI))
#define isdig(c) (ctable[c] & _DI)
#define isodigit(c) ((c) >= '0' && (c) <= '7')
#define isholl(c) (ctable[c] & _HO)

static char ctable[256] = {
    0,         /* nul */
    0,         /* soh */
    0,         /* stx */
    0,         /* etx */
    0,         /* eot */
    0,         /* enq */
    0,         /* ack */
    0,         /* bel */
    0,         /* bs  */
    _BL,       /* ht  */
    0,         /* nl  */
    _BL,       /* vt  */
    _BL,       /* np  */
    _BL,       /* cr  */
    0,         /* so  */
    0,         /* si  */
    0,         /* dle */
    0,         /* dc1 */
    0,         /* dc2 */
    0,         /* dc3 */
    0,         /* dc4 */
    0,         /* nak */
    0,         /* syn */
    0,         /* etb */
    0,         /* can */
    0,         /* em  */
    0,         /* sub */
    0,         /* esc */
    0,         /* fs  */
    0,         /* gs  */
    0,         /* rs  */
    0,         /* us  */
    _BL,       /* sp  */
    0,         /* !  */
    0,         /* "  */
    0,         /* #  */
    _CS,       /* $  */
    0,         /* %  */
    0,         /* &  */
    0,         /* '  */
    0,         /* (  */
    0,         /* )  */
    0,         /* *  */
    0,         /* +  */
    0,         /* ,  */
    0,         /* -  */
    0,         /* .  */
    0,         /* /  */
    _DI,       /* 0  */
    _DI,       /* 1  */
    _DI,       /* 2  */
    _DI,       /* 3  */
    _DI,       /* 4  */
    _DI,       /* 5  */
    _DI,       /* 6  */
    _DI,       /* 7  */
    _DI,       /* 8  */
    _DI,       /* 9  */
    0,         /* :  */
    0,         /* ;  */
    0,         /* <  */
    0,         /* =  */
    0,         /* >  */
    0,         /* ?  */
    0,         /* @  */
    _CS | _HD, /* A  */
    _CS | _HD, /* B  */
    _CS | _HD, /* C  */
    _CS | _HD, /* D  */
    _CS | _HD, /* E  */
    _CS | _HD, /* F  */
    _CS,       /* G  */
    _CS | _HO, /* H  */
    _CS,       /* I  */
    _CS,       /* J  */
    _CS,       /* K  */
    _CS,       /* L  */
    _CS,       /* M  */
    _CS,       /* N  */
    _CS,       /* O  */
    _CS,       /* P  */
    _CS,       /* Q  */
    _CS,       /* R  */
    _CS,       /* S  */
    _CS,       /* T  */
    _CS,       /* U  */
    _CS,       /* V  */
    _CS,       /* W  */
    _CS,       /* X  */
    _CS,       /* Y  */
    _CS,       /* Z  */
    0,         /* [  */
    0,         /* \  */
    0,         /* ]  */
    0,         /* ^  */
    _CS,       /* _  */
    0,         /* `  */
    _CS | _HD, /* a  */
    _CS | _HD, /* b  */
    _CS | _HD, /* c  */
    _CS | _HD, /* d  */
    _CS | _HD, /* e  */
    _CS | _HD, /* f  */
    _CS,       /* g  */
    _CS | _HO, /* h  */
    _CS,       /* i  */
    _CS,       /* j  */
    _CS,       /* k  */
    _CS,       /* l  */
    _CS,       /* m  */
    _CS,       /* n  */
    _CS,       /* o  */
    _CS,       /* p  */
    _CS,       /* q  */
    _CS,       /* r  */
    _CS,       /* s  */
    _CS,       /* t  */
    _CS,       /* u  */
    _CS,       /* v  */
    _CS,       /* w  */
    _CS,       /* x  */
    _CS,       /* y  */
    _CS,       /* z  */
    0,         /* {  */
    0,         /* |  */
    0,         /* }  */
    0,         /* ~  */
    0,         /* del */
};
