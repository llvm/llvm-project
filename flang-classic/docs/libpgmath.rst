*********
libpgmath
*********

Libpgmath implements decent subset of Fortran Mathematical Intrinsics. These are:

* acos

* asin

* atan

* atan2

* cos

* sin

* tan

* cosh

* sinh

* tanh

* exp

* log

* log10

* pow

* powi1 R{4, 8} * I4

* powi R{4, 8} * I4(:)

* powk1 R{4, 8} * I8

* powk R{4, 8} * I8(:)

* sincos

* div

* sqrt

* mod

* size

APIs
####

The library exposes two APIs:

* **mth API** (a.k.a old one as it was also present in pre-libpgmath runtime library), implemented by generic part of libpgmath, mostly as wrappers calling mathematical functions from underlying mathematical library (libm or libamath).

* **frp API** (fast-relaxed-precise), exposing two kinds of symbols:
    
    * jump table proxies (defined in ``mth_*defs.c`` files) exposing functions utilizing architecture-specific dispatch jump tables; in case of serial functions, usually the jumps are aimed at the generic functions exposed by ``mth API``; in case of vectorized functions (or functions accepting arguments of complex number type), the jumps are aimed at specific implementation functions exposed by ``frp API``,
 
    * implementation functions (vectorized or handling complex numbers type arguments); most of them are just wrappers implemented as 'for' loops calling mathematical functions from underlying mathematical library (libm or libamath).

By default, for every serial function Flang frontend generates calls to ``frp API`` functions (jump table proxies) defined in ``mth_128defs.c`` file. All those functions are implemented as calls through the jump table. When ``-My,164,0x800000`` flag is used, Flang frontend generates calls to ``mth API`` functions overriding jump table (this also prevents use of vectorized functions!). 

The ``mth_128defs_init.c`` file contains ``math_tables`` initialization for every ``frp API`` proxy function. All those functions are ``_init`` suffixed. It is called at first execution of any of proxy function when libpgmath is linked as a static library (when the Fortran application is linked against shared libpgmath library, ``__mth_dispatch()`` function is called by
shared object constructor (``ctor``) before ``main()`` function is executed). 

The ``dispatch.c`` file instantiates ``mth_tables``. It contains calls to all of the files refered from mth_tables.

The APIs also differ by the prefixes and suffixes they use. ``frp API`` functions look like this:

::
    
     __{gfrp}{sdcz}_<NAME>_<VL>[_{init|prof}]

Where:

::

    g - generic
    f - fast
    r - relaxed
    p - precise
    
    s - single
    d - double
    c - single complex
    z - double complex

``mth API`` function will have the following prefixes:

* ``__mth_i_`` - math intrinsic
    * Vector ABI prefixes e.g. ``__ZGVxN1v_mth_i_``
* ``__pmth_i_``
* ``__utl_i_``

For vectorized functions, LLVM's ``LoopVectorize`` pass utilizes ``veclib`` mapping from serial ``frp API`` function calls to corresponding vectorized functions. There's a PGI's map (PGMATH) available, which aims functions defined in ``mth_128defs.c`` file on AArch64 and Power architectures and in ``mth_256defs.c`` and ``mth_512defs.c`` files for Intel architectures,

Math table entries
##################

The MTHINTRIN macros
********************

The ``MTHINTRIN`` macro is defined (differently) in two places: ``lib/common/mth_tbldefs.h`` and ``lib/common/dispatch.c`` (preceded by ``'#undef MTHINTRIN'`` to get rid of definition provided by ``mth_tbldefs.h``). Both definitions take the same number of arguments and are used while pre-processing ``mth_*.h`` files of selected ``math_tables`` (note that ``dispatch.c`` does include ``those mth_*.h`` files twice: before and after ``'#undef MTHINTRIN'`` in order to process them with both of the ``MTHINTRIN`` macro definitions).

In currently used ``math_tables`` (both generic and x86_64) the last parameter (name for Sleef) is always given a pointer to ``__math_dispatch_error()`` function.

Porting
#######

There are currently two ports:

* AArch64 port in ``lib/aarch64`` directory, contributed by Cavium

* x86_64 port in ``lib/x86_64`` directory, contributed by Nvidia

Every port can override ``math_tables`` to point to different function names, however currently only the x86_64 port makes use of this possibility.

The use of overridden math_tables must be signaled in the given port CMakeLists.txt file:

::

    add_subdirectory("math_tables")

The ``math_tables`` directory should contain header files, one for each of mathematical function implemented in libpgmath. List of all of the exposed headers should be provided in ``CMakeLists.txt`` file held in given ``math_tables`` directory:

::

    set(SRCS
      mth_acosdefs.h
      mth_asindefs.h
      mth_atandefs.h
      mth_atan2defs.h
      mth_cosdefs.h
      mth_sindefs.h
      mth_tandefs.h
      mth_coshdefs.h
      mth_sinhdefs.h
      mth_tanhdefs.h
      mth_expdefs.h
      mth_logdefs.h
      mth_log10defs.h
      mth_powdefs.h
      mth_powidefs.h
      mth_sincosdefs.h
      mth_divdefs.h
      mth_sqrtdefs.h
      mth_moddefs.h)

    set(NEW_SRCS)
    foreach(file ${SRCS})
      list(APPEND NEW_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    endforeach()
    set(SRCS ${NEW_SRCS})
    set(DEPENDENCIES "${SRCS}")
    
    string(REPLACE ";" " -D" DEFINITIONS "${DEFINITIONS}")
    set(DEFINITIONS "-D${DEFINITIONS}")
    string(REPLACE ";" " " SRCS "${SRCS}")
    list(APPEND PREPROCESSOR "${CMAKE_C_COMPILER} -E ${DEFINITIONS} -DPGFLANG ${FLAGS} ${SRCS}")
    separate_arguments(PREPROCESSOR UNIX_COMMAND "${PREPROCESSOR}"


