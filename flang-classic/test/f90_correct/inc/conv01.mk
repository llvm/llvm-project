#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test chartoq  ########


conv01: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/conv01.f08 fcheck.$(OBJX)
	-$(RM) conv01.$(EXESUFFIX) conv01.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/conv01.f08 -o conv01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) conv01.$(OBJX) fcheck.$(OBJX) $(LIBS) -o conv01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test conv01
	conv01.$(EXESUFFIX)

verify: ;

conv01.run: run

