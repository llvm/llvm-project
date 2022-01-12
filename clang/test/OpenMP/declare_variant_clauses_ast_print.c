//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -ast-print -o - %s | FileCheck %s --check-prefix=PRINT

//RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fopenmp -fopenmp-version=51 \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses -DWIN -fms-compatibility \
//RUN:   -ast-print -o - %s | FileCheck %s --check-prefixes=PRINT,PRINTW

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -ast-dump -o - %s | FileCheck %s --check-prefix=DUMP

//RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fopenmp -fopenmp-version=51 \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses -DWIN -fms-compatibility \
//RUN:   -ast-dump -o - %s | FileCheck %s --check-prefixes=DUMP,DUMPW

typedef void *omp_interop_t;

#ifdef WIN
//DUMPW: FunctionDecl{{.*}}win_foov
//PRINTW: void win_foov(int n, double *y, void *interop_obj);

void win_foov(int n, double *y, void *interop_obj);

//DUMPW: FunctionDecl{{.*}}win_foo
//DUMPW: OMPDeclareVariantAttr
//DUMPW-NEXT: DeclRefExpr{{.*}}win_foov
//PRINTW: #pragma omp declare variant(win_foov) match(construct={dispatch}, device={arch(x86_64)}) append_args(interop(targetsync))
//PRINTW: void win_foo(int n, double *y);

#pragma omp declare variant (win_foov) \
  match(construct={dispatch}, device={arch(x86_64)}) \
  append_args(interop(targetsync))
void _cdecl win_foo(int n, double *y);
#endif // WIN

//DUMP: FunctionDecl{{.*}}c_foov
//PRINT: void c_foov(int n, double *y, void *interop_obj);

void c_foov(int n, double *y, void *interop_obj);

//DUMP: FunctionDecl{{.*}}c_foo
//DUMP: OMPDeclareVariantAttr
//DUMP-NEXT: DeclRefExpr{{.*}}c_foov
//PRINT: #pragma omp declare variant(c_foov) match(construct={dispatch}, device={arch(x86_64)}) append_args(interop(targetsync))
//PRINT: void c_foo(int n, double *y);

#pragma omp declare variant (c_foov) \
  match(construct={dispatch}, device={arch(x86_64)}) \
  append_args(interop(targetsync))
void c_foo(int n, double *y);
