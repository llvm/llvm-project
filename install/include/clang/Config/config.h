/* This generated file is for internal use. Do not include it from headers. */

#ifdef CLANG_CONFIG_H
#error config.h can only be included once
#else
#define CLANG_CONFIG_H

/* Bug report URL. */
#define BUG_REPORT_URL "https://github.com/llvm/llvm-project/issues/"

/* Default to -fPIE and -pie on Linux. */
#define CLANG_DEFAULT_PIE_ON_LINUX 1

/* Default linker to use. */
#define CLANG_DEFAULT_LINKER ""

/* Default C++ stdlib to use. */
#define CLANG_DEFAULT_CXX_STDLIB ""

/* Default runtime library to use. */
#define CLANG_DEFAULT_RTLIB ""

/* Default unwind library to use. */
#define CLANG_DEFAULT_UNWINDLIB ""

/* Default objcopy to use */
#define CLANG_DEFAULT_OBJCOPY "objcopy"

/* Default OpenMP runtime used by -fopenmp. */
#define CLANG_DEFAULT_OPENMP_RUNTIME "libomp"

/* Default architecture for SystemZ. */
#define CLANG_SYSTEMZ_DEFAULT_ARCH "z10"

/* Multilib basename for libdir. */
#define CLANG_INSTALL_LIBDIR_BASENAME "lib"

/* Relative directory for resource files */
#define CLANG_RESOURCE_DIR ""

/* Directories clang will search for headers */
#define C_INCLUDE_DIRS ""

/* Directories clang will search for configuration files */
/* #undef CLANG_CONFIG_FILE_SYSTEM_DIR */
/* #undef CLANG_CONFIG_FILE_USER_DIR */

/* Default <path> to all compiler invocations for --sysroot=<path>. */
#define DEFAULT_SYSROOT ""

/* Directory where gcc is installed. */
#define GCC_INSTALL_PREFIX ""

/* Define if we have libxml2 */
#define CLANG_HAVE_LIBXML 1

/* Define if we have sys/resource.h (rlimits) */
#define CLANG_HAVE_RLIMITS 1

/* Define if we have dlfcn.h */
#define CLANG_HAVE_DLFCN_H 1

/* Define if dladdr() is available on this platform. */
#define CLANG_HAVE_DLADDR 1

/* Linker version detected at compile time. */
/* #undef HOST_LINK_VERSION */

/* pass --build-id to ld */
/* #undef ENABLE_LINKER_BUILD_ID */

/* enable x86 relax relocations by default */
#define ENABLE_X86_RELAX_RELOCATIONS 1

/* Enable IEEE binary128 as default long double format on PowerPC Linux. */
#define PPC_LINUX_DEFAULT_IEEELONGDOUBLE 0

/* Enable each functionality of modules */
#define CLANG_ENABLE_OBJC_REWRITER 0
#define CLANG_ENABLE_STATIC_ANALYZER 1

/* Spawn a new process clang.exe for the CC1 tool invocation, when necessary */
#define CLANG_SPAWN_CC1 0

/* Whether CIR is built into Clang */
#define CLANG_ENABLE_CIR 0

#endif
