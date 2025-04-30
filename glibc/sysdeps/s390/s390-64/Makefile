ifeq ($(subdir),gmon)
sysdep_routines += s390x-mcount
endif

ifeq ($(subdir),elf)
CFLAGS-rtld.c += -Wno-uninitialized -Wno-unused
CFLAGS-dl-load.c += -Wno-unused
CFLAGS-dl-reloc.c += -Wno-unused

$(objpfx)tst-glibc-hwcaps: $(objpfx)libmarkermod2-1.so \
  $(objpfx)libmarkermod3-1.so $(objpfx)libmarkermod4-1.so
$(objpfx)tst-glibc-hwcaps.out: \
  $(objpfx)libmarkermod2.so \
    $(objpfx)glibc-hwcaps/z13/libmarkermod2.so \
  $(objpfx)libmarkermod3.so \
    $(objpfx)glibc-hwcaps/z13/libmarkermod3.so \
    $(objpfx)glibc-hwcaps/z14/libmarkermod3.so \
  $(objpfx)libmarkermod4.so \
    $(objpfx)glibc-hwcaps/z13/libmarkermod4.so \
    $(objpfx)glibc-hwcaps/z14/libmarkermod4.so \
    $(objpfx)glibc-hwcaps/z15/libmarkermod4.so \

$(objpfx)glibc-hwcaps/z13/libmarkermod2.so: $(objpfx)libmarkermod2-2.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/z13/libmarkermod3.so: $(objpfx)libmarkermod3-2.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/z14/libmarkermod3.so: $(objpfx)libmarkermod3-3.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/z13/libmarkermod4.so: $(objpfx)libmarkermod4-2.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/z14/libmarkermod4.so: $(objpfx)libmarkermod4-3.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/z15/libmarkermod4.so: $(objpfx)libmarkermod4-4.so
	$(make-target-directory)
	cp $< $@

ifeq (no,$(build-hardcoded-path-in-tests))
# This is an ld.so.cache test, and RPATH/RUNPATH in the executable
# interferes with its test objectives.
tests-container += tst-glibc-hwcaps-cache
endif

endif # $(subdir) == elf
