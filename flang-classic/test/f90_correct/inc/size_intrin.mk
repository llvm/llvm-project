# Call to "size" intrinsic is inlined
#
########## Make rule for test size_intrin.f90 ########
nearest_intrin: run

build:  $(SRC)/size_intrin.f90
	-$(RM) size_intrin.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/size_intrin.f90 -o size_intrin.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) size_intrin.$(OBJX) check.$(OBJX) $(LIBS) -o size_intrin.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test size_intrin
	size_intrin.$(EXESUFFIX)

verify: ;

