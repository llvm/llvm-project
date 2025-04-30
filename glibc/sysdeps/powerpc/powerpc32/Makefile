# Powerpc32 specific build options.

# Some Powerpc32 variants assume soft-fp is the default even though there is
# an fp variant so provide -mhard-float if --with-fp is explicitly passed.

ifeq ($(with-fp),yes)
+cflags += -mhard-float
ASFLAGS += -mhard-float
sysdep-LDFLAGS += -mhard-float
endif

ifeq ($(subdir),gmon)
sysdep_routines += ppc-mcount compat-ppc-mcount
static-only-routines += ppc-mcount
shared-only-routines += compat-ppc-mcount
endif

ifeq ($(subdir),misc)
sysdep_routines += gprsave0 gprrest0 gprsave1 gprrest1
endif

# On PPC, -fpic works until the GOT contains 32768 bytes, and possibly
# more depending on how clever the linker is.  Each GOT entry takes 4 bytes,
# so that's at least 8192 entries.  Since libc only uses about 2000 entries,
# we want to use -fpic, because this generates fewer relocs.
ifeq (yes,$(build-shared))
pic-ccflag = -fpic
endif

ifeq ($(subdir),csu)
# There is no benefit to using sdata for these objects, and the user
# of the library should be able to control what goes into sdata.
CFLAGS-init.o = -G0
CFLAGS-gmon-start.o = -G0

ifeq (yes,$(build-shared))
# Compatibility
ifeq (yes,$(have-protected))
CPPFLAGS-libgcc-compat.S = -DHAVE_DOT_HIDDEN
endif
sysdep_routines += libgcc-compat
shared-only-routines += libgcc-compat
endif
endif

ifeq ($(subdir),elf)
# extra shared linker files to link only into dl-allobjs.so
sysdep-rtld-routines += dl-start
endif
