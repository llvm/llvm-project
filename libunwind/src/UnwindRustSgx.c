//===--------------------- UnwindRustSgx.c ----------------------------------===//
//
////                     The LLVM Compiler Infrastructure
////
//// This file is dual licensed under the MIT and the University of Illinois Open
//// Source Licenses. See LICENSE.TXT for details.
////
////
////===----------------------------------------------------------------------===//

#define _GNU_SOURCE
#include <link.h>

#include <elf.h>
#include <stdarg.h>
#include <stdio.h>
#include <stddef.h>
#include "UnwindRustSgx.h"


#define max_log 256


__attribute__((weak)) struct _IO_FILE *stderr  = (_IO_FILE *)-1;

static int vwrite_err(const char *format, va_list ap)
{
    char s[max_log];
    int len = 0;
    s[0]='\0';
    len = vsnprintf(s, max_log, format, ap);
    __rust_print_err((uint8_t *)s, len);
    return len;
}

static int write_err(const char *format, ...)
{
    int ret;
    va_list args;
        va_start(args, format);
    ret = vwrite_err(format, args);
    va_end(args);


    return ret;
}

__attribute__((weak)) int fprintf (FILE *__restrict __stream,
            const char *__restrict __format, ...)
{

    int ret;
    if (__stream != stderr) {
        write_err("Rust SGX Unwind supports only writing to stderr\n");
        return -1;
    } else {
        va_list args;
        ret = 0;
        va_start(args, __format);
        ret += vwrite_err(__format, args);
        va_end(args);
    }

    return ret;
}

__attribute__((weak)) int fflush (FILE *__stream)
{
    // We do not need to do anything here.
    return 0;
}




__attribute__((weak)) void __assert_fail(const char * assertion,
                                       const char * file,
                           unsigned int line,
                           const char * function)
{
    write_err("%s:%d %s %s\n", file, line, function, assertion);
    abort();
}



// We do not report stack over flow detected.
// Calling write_err uses more stack due to the way we have implemented it.
// With possible enabling of stack probes, we should not
// get into __stack_chk_fail() at all.
__attribute__((weak))  void __stack_chk_fail() {
    abort();
}

/*
 * Below are defined for all executibles compiled for
 * x86_64-fortanix-unknown-sgx rust target.
 * Ref: rust/src/libstd/sys/sgx/abi/entry.S
 */

extern uint64_t TEXT_BASE;
extern uint64_t TEXT_SIZE;
extern uint64_t EH_FRM_HDR_BASE;
extern uint64_t EH_FRM_HDR_SIZE;
extern char IMAGE_BASE;

typedef Elf64_Phdr Elf_Phdr; 
int
dl_iterate_phdr (int (*callback) (struct dl_phdr_info *,
                  size_t, void *),
         void *data)
{
    struct dl_phdr_info info;
    struct dl_phdr_info *pinfo = &info;
    Elf_Phdr phdr[2];
    int ret = 0;


    size_t text_size = TEXT_SIZE;
    size_t eh_base_size = EH_FRM_HDR_SIZE;

    memset(pinfo, 0, sizeof(*pinfo));

    pinfo->dlpi_addr = (ElfW(Addr))&IMAGE_BASE;
    pinfo->dlpi_phnum = 2;

    pinfo->dlpi_phdr = phdr;
    memset(phdr, 0, 2*sizeof(*phdr));


    phdr[0].p_type = PT_LOAD;
    phdr[0].p_vaddr = (size_t)TEXT_BASE;
    phdr[0].p_memsz = text_size;

    phdr[1].p_type = PT_GNU_EH_FRAME;
    phdr[1].p_vaddr = (size_t)EH_FRM_HDR_BASE;
    phdr[1].p_memsz = eh_base_size;


    ret = callback (&info, sizeof (struct dl_phdr_info), data);
    return ret;
}

struct libwu_rs_alloc_meta {
    size_t alloc_size;
    // Should we put a signatre guard before ptr for oob access?
    unsigned char ptr[0];
};

#define META_FROM_PTR(__PTR) (struct libwu_rs_alloc_meta *) \
    ((unsigned char *)__PTR - offsetof(struct libwu_rs_alloc_meta, ptr))

void *libuw_malloc(size_t size)
{
    struct libwu_rs_alloc_meta *meta;
    size_t alloc_size = size + sizeof(struct libwu_rs_alloc_meta);
    meta = (void *)__rust_c_alloc(alloc_size, sizeof(size_t));
    if (!meta) {
        return NULL;
    }
    meta->alloc_size = alloc_size;
    return (void *)meta->ptr;
}

void libuw_free(void *p)
{
    struct libwu_rs_alloc_meta *meta;
    if (!p) {
        return;
    }
    meta = META_FROM_PTR(p);
    __rust_c_dealloc((unsigned char *)meta, meta->alloc_size, sizeof(size_t));
}
