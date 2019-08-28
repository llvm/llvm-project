/* This generated file is for internal use. Do not include it from headers. */

#ifdef CLANG_CONFIG_H
#error config.h can only be included once
#else
#define CLANG_CONFIG_H

/* Bug report URL. */
#define BUG_REPORT_URL "${BUG_REPORT_URL}"

/* Default linker to use. */
#define CLANG_DEFAULT_LINKER "${CLANG_DEFAULT_LINKER}"

/* Default C/ObjC standard to use. */
#cmakedefine CLANG_DEFAULT_STD_C LangStandard::lang_${CLANG_DEFAULT_STD_C}

/* Default C++/ObjC++ standard to use. */
#cmakedefine CLANG_DEFAULT_STD_CXX LangStandard::lang_${CLANG_DEFAULT_STD_CXX}

/* Default C++ stdlib to use. */
#define CLANG_DEFAULT_CXX_STDLIB "${CLANG_DEFAULT_CXX_STDLIB}"

/* Default runtime library to use. */
#define CLANG_DEFAULT_RTLIB "${CLANG_DEFAULT_RTLIB}"

/* Default unwind library to use. */
#define CLANG_DEFAULT_UNWINDLIB "${CLANG_DEFAULT_UNWINDLIB}"

/* Default objcopy to use */
#define CLANG_DEFAULT_OBJCOPY "${CLANG_DEFAULT_OBJCOPY}"

/* Default OpenMP runtime used by -fopenmp. */
#define CLANG_DEFAULT_OPENMP_RUNTIME "${CLANG_DEFAULT_OPENMP_RUNTIME}"

/* Default architecture for OpenMP offloading to Nvidia GPUs. */
#define CLANG_OPENMP_NVPTX_DEFAULT_ARCH "${CLANG_OPENMP_NVPTX_DEFAULT_ARCH}"

/* Default Tapir runtime used by -ftapir. */
#define CLANG_DEFAULT_TAPIR_RUNTIME "${CLANG_DEFAULT_TAPIR_RUNTIME}"

/* Multilib suffix for libdir. */
#define CLANG_LIBDIR_SUFFIX "${CLANG_LIBDIR_SUFFIX}"

/* Relative directory for resource files */
#define CLANG_RESOURCE_DIR "${CLANG_RESOURCE_DIR}"

/* Directories clang will search for headers */
#define C_INCLUDE_DIRS "${C_INCLUDE_DIRS}"

/* Directories clang will search for configuration files */
#cmakedefine CLANG_CONFIG_FILE_SYSTEM_DIR "${CLANG_CONFIG_FILE_SYSTEM_DIR}"
#cmakedefine CLANG_CONFIG_FILE_USER_DIR "${CLANG_CONFIG_FILE_USER_DIR}"

/* Default <path> to all compiler invocations for --sysroot=<path>. */
#define DEFAULT_SYSROOT "${DEFAULT_SYSROOT}"

/* Directory where gcc is installed. */
#define GCC_INSTALL_PREFIX "${GCC_INSTALL_PREFIX}"

/* Define if we have libxml2 */
#cmakedefine CLANG_HAVE_LIBXML ${CLANG_HAVE_LIBXML}

/* Define if we have sys/resource.h (rlimits) */
#cmakedefine CLANG_HAVE_RLIMITS ${CLANG_HAVE_RLIMITS}

/* The LLVM product name and version */
#define BACKEND_PACKAGE_STRING "${BACKEND_PACKAGE_STRING}"

/* Linker version detected at compile time. */
#cmakedefine HOST_LINK_VERSION "${HOST_LINK_VERSION}"

/* pass --build-id to ld */
#cmakedefine ENABLE_LINKER_BUILD_ID

/* enable x86 relax relocations by default */
#cmakedefine01 ENABLE_X86_RELAX_RELOCATIONS

/* Enable the experimental new pass manager by default */
#cmakedefine01 ENABLE_EXPERIMENTAL_NEW_PASS_MANAGER

/* Enable each functionality of modules */
#cmakedefine01 CLANG_ENABLE_ARCMT
#cmakedefine01 CLANG_ENABLE_OBJC_REWRITER
#cmakedefine01 CLANG_ENABLE_STATIC_ANALYZER


/* ===== kitsune-centric settings  */
#cmakedefine01 KITSUNE_ENABLE_KOKKOS 
#define KITSUNE_KOKKOS_INCLUDE_DIR       "${KOKKOS_INCLUDE_DIR}"
#define KITSUNE_KOKKOS_LIBRARY_DIR       "${KOKKOS_LIBRARY_DIR}"
#define KITSUNE_KOKKOS_LINK_LIBS         "${KOKKOS_LINK_LIBS}"

#cmakedefine01 KITSUNE_ENABLE_CILKRTS 
#define KITSUNE_CILKRTS_LIBRARY_DIR      "${CILKRTS_LIBRARY_DIR}"
#define KITSUNE_CILKRTS_LINK_LIBS        "${CILKRTS_LINK_LIBS}"

#cmakedefine01 KITSUNE_ENABLE_QTHREADS   
#define KITSUNE_QTHREADS_LIBRARY_DIR     "${QTHREADS_LIBRARY_DIR}"
#define KITSUNE_QTHREADS_LINK_LIBS       "${QTHREADS_LINK_LIBS}"

#cmakedefine01 KITSUNE_ENABLE_REALM
#define KITSUNE_REALM_LIBRARY_DIR        "${REALM_LIBRARY_DIR}"
#define KITSUNE_REALM_LINK_LIBS          "${REALM_LINK_LIBS}"

#cmakedefine01 KITSUNE_ENABLE_OPENMP
#define KITSUNE_OPENMP_LIBRARY_DIR        "${OPENMP_LIBRARY_DIR}"
#define KITSUNE_OPENMP_LINK_LIBS          "${OPENMP_LINK_LIBS}"

#cmakedefine01 KITSUNE_ENABLE_CUDA
#define KITSUNE_CUDA_LIBRARY_DIR         "${CUDA_LIBRARY_DIR}"
#define KITSUNE_CUDA_LINK_LIBS           "${CUDA_LINK_LIBS}"

#endif
