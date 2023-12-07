//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <stdio.h>

#include <stdio.h>
#include <stdarg.h>

#include "test_macros.h"

#ifdef getc
#error getc is defined
#endif

#ifdef putc
#error putc is defined
#endif

#ifdef clearerr
#error clearerr is defined
#endif

#ifdef feof
#error feof is defined
#endif

#ifdef ferror
#error ferror is defined
#endif

#ifndef BUFSIZ
#error BUFSIZ not defined
#endif

#ifndef EOF
#error EOF not defined
#endif

#ifndef FILENAME_MAX
#error FILENAME_MAX not defined
#endif

#ifndef FOPEN_MAX
#error FOPEN_MAX not defined
#endif

#ifndef L_tmpnam
#error L_tmpnam not defined
#endif

#ifndef NULL
#error NULL not defined
#endif

#ifndef SEEK_CUR
#error SEEK_CUR not defined
#endif

#ifndef SEEK_END
#error SEEK_END not defined
#endif

#ifndef SEEK_SET
#error SEEK_SET not defined
#endif

#ifndef TMP_MAX
#error TMP_MAX not defined
#endif

#ifndef _IOFBF
#error _IOFBF not defined
#endif

#ifndef _IOLBF
#error _IOLBF not defined
#endif

#ifndef _IONBF
#error _IONBF not defined
#endif

#ifndef stderr
#error stderr not defined
#endif

#ifndef stdin
#error stdin not defined
#endif

#ifndef stdout
#error stdout not defined
#endif

TEST_CLANG_DIAGNOSTIC_IGNORED("-Wformat-zero-length")
TEST_GCC_DIAGNOSTIC_IGNORED("-Wformat-zero-length")

FILE* fp = 0;
fpos_t fpos = fpos_t();
size_t s = 0;
char* cp = 0;
char arr[] = {'a', 'b'};
va_list va;
ASSERT_SAME_TYPE(int,    decltype(remove("")));
ASSERT_SAME_TYPE(int,    decltype(rename("","")));
ASSERT_SAME_TYPE(FILE*,  decltype(tmpfile()));
TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
TEST_GCC_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
ASSERT_SAME_TYPE(char*,  decltype(tmpnam(cp)));
TEST_DIAGNOSTIC_POP
ASSERT_SAME_TYPE(int,    decltype(fclose(fp)));
ASSERT_SAME_TYPE(int,    decltype(fflush(fp)));
ASSERT_SAME_TYPE(FILE*,  decltype(fopen("", "")));
ASSERT_SAME_TYPE(FILE*,  decltype(freopen("", "", fp)));
ASSERT_SAME_TYPE(void,   decltype(setbuf(fp,cp)));
ASSERT_SAME_TYPE(int,    decltype(vfprintf(fp,"",va)));
ASSERT_SAME_TYPE(int,    decltype(fprintf(fp," ")));
ASSERT_SAME_TYPE(int,    decltype(fscanf(fp,"")));
ASSERT_SAME_TYPE(int,    decltype(printf("\n")));
ASSERT_SAME_TYPE(int,    decltype(scanf("\n")));
ASSERT_SAME_TYPE(int,    decltype(snprintf(cp,0,"p")));
TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
TEST_GCC_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
ASSERT_SAME_TYPE(int,    decltype(sprintf(cp," ")));
TEST_DIAGNOSTIC_POP
ASSERT_SAME_TYPE(int,    decltype(sscanf("","")));
ASSERT_SAME_TYPE(int,    decltype(vfprintf(fp,"",va)));
ASSERT_SAME_TYPE(int,    decltype(vfscanf(fp,"",va)));
ASSERT_SAME_TYPE(int,    decltype(vprintf(" ",va)));
ASSERT_SAME_TYPE(int,    decltype(vscanf("",va)));
ASSERT_SAME_TYPE(int,    decltype(vsnprintf(cp,0," ",va)));
TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
TEST_GCC_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
ASSERT_SAME_TYPE(int,    decltype(vsprintf(cp," ",va)));
TEST_DIAGNOSTIC_POP
ASSERT_SAME_TYPE(int,    decltype(vsscanf("","",va)));
ASSERT_SAME_TYPE(int,    decltype(fgetc(fp)));
ASSERT_SAME_TYPE(char*,  decltype(fgets(cp,0,fp)));
ASSERT_SAME_TYPE(int,    decltype(fputc(0,fp)));
ASSERT_SAME_TYPE(int,    decltype(fputs("",fp)));
ASSERT_SAME_TYPE(int,    decltype(getc(fp)));
ASSERT_SAME_TYPE(int,    decltype(getchar()));
#if TEST_STD_VER < 14
TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
TEST_GCC_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
ASSERT_SAME_TYPE(char*,  decltype(gets(cp)));
TEST_DIAGNOSTIC_POP
#endif
ASSERT_SAME_TYPE(int,    decltype(putc(0,fp)));
ASSERT_SAME_TYPE(int,    decltype(putchar(0)));
ASSERT_SAME_TYPE(int,    decltype(puts("")));
ASSERT_SAME_TYPE(int,    decltype(ungetc(0,fp)));
ASSERT_SAME_TYPE(size_t, decltype(fread((void*)0,0,0,fp)));
ASSERT_SAME_TYPE(size_t, decltype(fwrite((const void*)arr,1,0,fp)));
#ifndef TEST_HAS_NO_FGETPOS_FSETPOS
ASSERT_SAME_TYPE(int,    decltype(fgetpos(fp, &fpos)));
#endif
ASSERT_SAME_TYPE(int,    decltype(fseek(fp, 0,0)));
#ifndef TEST_HAS_NO_FGETPOS_FSETPOS
ASSERT_SAME_TYPE(int,    decltype(fsetpos(fp, &fpos)));
#endif
ASSERT_SAME_TYPE(long,   decltype(ftell(fp)));
ASSERT_SAME_TYPE(void,   decltype(rewind(fp)));
ASSERT_SAME_TYPE(void,   decltype(clearerr(fp)));
ASSERT_SAME_TYPE(int,    decltype(feof(fp)));
ASSERT_SAME_TYPE(int,    decltype(ferror(fp)));
ASSERT_SAME_TYPE(void,   decltype(perror("")));
