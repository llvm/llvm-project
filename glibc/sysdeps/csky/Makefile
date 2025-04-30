ifeq ($(subdir),csu)
gen-as-const-headers += tcb-offsets.sym
endif

ASFLAGS-.os += $(pic-ccflag)

ifeq ($(subdir),gmon)
sysdep_routines += csky-mcount
endif
